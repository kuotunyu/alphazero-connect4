"""Training loop: self-play -> replay buffer -> optimize -> arena -> checkpoint.

Run locally:   python -m az.train --preset smoke
Run on Colab:  python -m az.train --preset full --ckpt-dir <drive dir> --resume

Self-play always uses the LATEST network; gating (>55% vs current best over
the arena games) only decides which weights are published as ``best.pt``.
Checkpoints are written atomically with a last/prev rotation so an
interrupted write never destroys the run; ``--resume`` picks up from
``ckpt_last.pt`` (falling back to ``ckpt_prev.pt``) if present.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from az.arena import MCTSAgent, PolicyAgent, RandomAgent, play_match
from az.config import PRESETS, Config
from az.elo import mle_rating
from az.mcts import RolloutEvaluator
from az.model import MASK_VALUE, NetEvaluator, PolicyValueNet
from az.replay import ReplayBuffer
from az.selfplay import play_games
from az.viz import plot_elo

CSV_FIELDS = [
    "iteration", "wallclock_s", "games", "avg_moves", "buffer_size",
    "loss_policy", "loss_value", "entropy_policy",
    "wr_random", "wr_policy_random", "wr_mcts200", "wr_best", "elo", "gated",
]


def train_steps(model, optimizer, buffer, cfg: Config, device, rng):
    """One iteration's optimisation pass. Returns (loss_p, loss_v, entropy)."""
    model.train()
    tot_p = tot_v = tot_e = 0.0
    for _ in range(cfg.steps_per_iter):
        planes, pis, zs = buffer.sample(cfg.batch_size, rng)
        x = torch.from_numpy(planes).to(device)
        target_pi = torch.from_numpy(pis).to(device)
        target_z = torch.from_numpy(zs).to(device)

        logits, v = model(x)
        # a column is legal iff its top cell (row 0, top-first planes) is empty
        legal = (x[:, 0, 0, :] + x[:, 1, 0, :]) < 0.5
        logits = logits.masked_fill(~legal, MASK_VALUE)
        logp = torch.log_softmax(logits, dim=1)
        loss_p = -(target_pi * logp).sum(dim=1).mean()
        loss_v = torch.nn.functional.mse_loss(v, target_z)
        loss = loss_p + loss_v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            entropy = -(logp.exp() * logp).sum(dim=1).mean()
        tot_p += loss_p.item()
        tot_v += loss_v.item()
        tot_e += entropy.item()
    n = cfg.steps_per_iter
    return tot_p / n, tot_v / n, tot_e / n


def evaluate(model, best_model, cfg: Config, device, rng) -> dict:
    """Arena vs the three anchors + the raw-policy learning probe."""
    net_eval = NetEvaluator(model, device)
    agent = MCTSAgent(net_eval, cfg.arena_sims, cfg.c_puct, cfg.arena_temp_moves)

    wr_random = play_match(agent, RandomAgent(), cfg.arena_games, rng).score
    wr_policy = play_match(PolicyAgent(net_eval), RandomAgent(),
                           cfg.raw_policy_games, rng).score
    anchor = MCTSAgent(RolloutEvaluator(rng), cfg.anchor_sims, cfg.c_puct,
                       cfg.arena_temp_moves)
    wr_anchor = play_match(agent, anchor, cfg.arena_games, rng).score
    best_agent = MCTSAgent(NetEvaluator(best_model, device), cfg.arena_sims,
                           cfg.c_puct, cfg.arena_temp_moves)
    wr_best = play_match(agent, best_agent, cfg.arena_games, rng).score
    return {
        "wr_random": wr_random,
        "wr_policy_random": wr_policy,
        "wr_mcts200": wr_anchor,
        "wr_best": wr_best,
    }


def save_checkpoint(ckpt_dir: Path, payload: dict) -> None:
    tmp, last, prev = (ckpt_dir / n for n in
                       ("ckpt_tmp.pt", "ckpt_last.pt", "ckpt_prev.pt"))
    torch.save(payload, tmp)
    if last.exists():
        os.replace(last, prev)
    os.replace(tmp, last)


def load_checkpoint(ckpt_dir: Path):
    for name in ("ckpt_last.pt", "ckpt_prev.pt"):
        path = ckpt_dir / name
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu",
                                     weights_only=False)
                print(f"[resume] loaded {path}")
                return payload
            except Exception as exc:  # corrupted write mid-crash
                print(f"[resume] {path} unreadable ({exc}), trying fallback")
    return None


def write_csv(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from ckpt_last.pt if present")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="graceful stop after this wallclock budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = PRESETS[args.preset]
    if args.iterations is not None:
        cfg.iterations = args.iterations
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.ckpt_dir or Path("checkpoints") / args.preset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = PolicyValueNet(cfg.blocks, cfg.filters).to(device)
    best_model = PolicyValueNet(cfg.blocks, cfg.filters).to(device)
    best_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_size)
    history: list[dict] = []
    start_iter = 0

    if args.resume:
        payload = load_checkpoint(ckpt_dir)
        if payload is not None:
            saved = payload["config"]
            for key in ("blocks", "filters", "buffer_size"):
                if saved.get(key) != getattr(cfg, key):
                    raise SystemExit(
                        f"checkpoint config mismatch on {key}: "
                        f"{saved.get(key)} != {getattr(cfg, key)}"
                    )
            model.load_state_dict(payload["model"])
            best_model.load_state_dict(payload["best"])
            optimizer.load_state_dict(payload["optimizer"])
            buffer = ReplayBuffer.from_state_dict(payload["buffer"])
            history = payload["history"]
            start_iter = payload["iteration"]
            print(f"[resume] continuing at iteration {start_iter}, "
                  f"buffer={len(buffer)}")

    print(f"[train] preset={args.preset} device={device} "
          f"params={sum(p.numel() for p in model.parameters())} "
          f"iters {start_iter}..{cfg.iterations - 1} ckpt={ckpt_dir}",
          flush=True)

    session_start = time.time()
    base_wallclock = history[-1]["wallclock_s"] if history else 0.0

    try:
        for it in range(start_iter, cfg.iterations):
            rng = np.random.default_rng((args.seed, it))
            t0 = time.time()

            # 1) self-play with the latest network
            evaluator = NetEvaluator(model, device, autocast=cfg.autocast)
            result = play_games(
                evaluator, cfg.games_per_iter, cfg.sims,
                parallel=cfg.parallel, c_puct=cfg.c_puct,
                noise_frac=cfg.noise_frac, noise_alpha=cfg.noise_alpha,
                temp_moves=cfg.temp_moves, rng=rng,
            )
            buffer.add(result.samples)
            avg_moves = float(np.mean([len(m) for m in result.moves_per_game]))

            # 2) optimize
            loss_p, loss_v, entropy = train_steps(
                model, optimizer, buffer, cfg, device, rng)

            # 3) arena + gating + Elo
            model.eval()
            metrics = evaluate(model, best_model, cfg, device, rng)
            gated = metrics["wr_best"] > cfg.gate_threshold
            if gated:
                best_model.load_state_dict(model.state_dict())
                torch.save({"model": best_model.state_dict(),
                            "config": best_model.config()},
                           ckpt_dir / "best.pt")
            elo = mle_rating([
                (0.0, metrics["wr_random"], cfg.arena_games),
                (cfg.anchor_elo, metrics["wr_mcts200"], cfg.arena_games),
            ])

            # 4) bookkeeping
            row = {
                "iteration": it,
                "wallclock_s": round(base_wallclock + time.time() - session_start, 1),
                "games": cfg.games_per_iter,
                "avg_moves": round(avg_moves, 2),
                "buffer_size": len(buffer),
                "loss_policy": round(loss_p, 4),
                "loss_value": round(loss_v, 4),
                "entropy_policy": round(entropy, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
                "elo": round(elo, 1),
                "gated": int(gated),
            }
            history.append(row)
            write_csv(ckpt_dir / "elo.csv", history)
            save_checkpoint(ckpt_dir, {
                "iteration": it + 1,
                "model": model.state_dict(),
                "best": best_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "buffer": buffer.state_dict(),
                "history": history,
                "config": cfg.to_dict(),
            })
            print(f"[iter {it:3d}] {time.time() - t0:6.1f}s "
                  f"buf={len(buffer):6d} loss_p={loss_p:.3f} loss_v={loss_v:.3f} "
                  f"H={entropy:.2f} | rand={metrics['wr_random']:.2f} "
                  f"pol={metrics['wr_policy_random']:.2f} "
                  f"mcts200={metrics['wr_mcts200']:.2f} "
                  f"best={metrics['wr_best']:.2f} elo={elo:6.1f} "
                  f"{'GATED' if gated else ''}", flush=True)

            if args.max_hours is not None and (
                    time.time() - session_start) > args.max_hours * 3600:
                print(f"[train] wallclock budget reached, stopping", flush=True)
                break
    finally:
        if history:
            plot_elo(history, ckpt_dir / "elo_curve.png")
            print(f"[train] elo curve -> {ckpt_dir / 'elo_curve.png'}",
                  flush=True)


if __name__ == "__main__":
    main()
