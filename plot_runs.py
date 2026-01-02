import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_FILES = ["config.json", "train.csv", "eval.csv"]


def find_runs(run_root: Path) -> List[Path]:
    """Recursively find run directories that contain REQUIRED_FILES."""
    run_dirs = []
    for cfg_path in run_root.rglob("config.json"):
        run_dir = cfg_path.parent
        ok = True
        for fn in REQUIRED_FILES:
            if not (run_dir / fn).exists():
                ok = False
                break
        if ok:
            run_dirs.append(run_dir)
    # de-dup (in case of weird symlinks)
    run_dirs = sorted(set(run_dirs))
    return run_dirs


def safe_read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return pd.DataFrame()


def get_label(run_dir: Path, cfg: Dict, group_keys: Optional[List[str]] = None) -> str:
    """
    Produce a readable label.
    Prefer cfg keys if available; fall back to path parts.
    """
    exp = cfg.get("exp_name", None)
    seed = cfg.get("seed", None)
    run_id = cfg.get("run_id", None)

    if group_keys:
        parts = []
        for k in group_keys:
            if k in cfg:
                parts.append(f"{k}={cfg[k]}")
        if parts:
            base = ",".join(parts)
        else:
            base = exp or run_dir.name
    else:
        base = exp or run_dir.name

    suffix = []
    if seed is not None:
        suffix.append(f"seed={seed}")
    # if run_id is not None:
        # shorten if too long
        # rid = str(run_id)
        # suffix.append(f"id={rid[:8]}")
    # if suffix:
        # return base + " (" + ", ".join(suffix) + ")"
    return base


def group_id_from_cfg(cfg: Dict, group_keys: List[str]) -> str:
    """
    Build a stable group id string from config keys.
    Missing key -> 'NA'.
    """
    vals = []
    for k in group_keys:
        vals.append(f"{k}={cfg.get(k, 'NA')}")
    return "|".join(vals)


def compute_step_to_threshold(
    eval_df: pd.DataFrame,
    threshold: float,
    k: int,
    step_col: str = "global_step",
    win_col: str = "win_rate",
) -> Tuple[Optional[int], bool]:
    """
    Return (step_to_threshold, reached).
    Rule: first time win_rate >= threshold holds for k consecutive eval points.
    """
    if eval_df.empty or (win_col not in eval_df.columns) or (step_col not in eval_df.columns):
        return None, False

    df = eval_df[[step_col, win_col]].dropna().sort_values(step_col)
    if df.empty:
        return None, False

    meets = (df[win_col].values >= threshold).astype(int)
    # find first index where k consecutive ones
    consec = 0
    for i, m in enumerate(meets):
        consec = consec + 1 if m == 1 else 0
        if consec >= k:
            step = int(df.iloc[i][step_col])
            return step, True
    return None, False


def plot_raw_runs(
    runs: List[Dict],
    out_dir: Path,
    metric: str,
    x_col: str = "global_step",
    y_col: str = "",
    title: str = "",
    ylabel: str = "",
    filename: str = "",
):
    """
    Plot each run as one line (no grouping).
    runs: list of dicts with keys: label, df
    """
    plt.figure()
    for r in runs:
        df = r["df"]
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        d = df[[x_col, y_col]].dropna().sort_values(x_col)
        if d.empty:
            continue
        plt.plot(d[x_col].values, d[y_col].values, label=r["label"])
    plt.xlabel(x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or metric)
    if len(runs) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grouped_runs(
    runs: List[Dict],
    out_dir: Path,
    metric: str,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    filename: str,
    grid_step: int = 100,
):
    """
    Group runs by group_id and plot mean +/- std across runs (seeds).
    Uses interpolation to a common x grid for each group.
    """
    # group
    groups: Dict[str, List[Dict]] = {}
    for r in runs:
        groups.setdefault(r["group_id"], []).append(r)

    plt.figure()
    for gid, items in groups.items():
        # collect per-run curves
        curves = []
        x_min, x_max = None, None
        for it in items:
            df = it["df"]
            if df.empty or x_col not in df.columns or y_col not in df.columns:
                continue
            d = df[[x_col, y_col]].dropna().sort_values(x_col)
            if d.empty:
                continue
            x = d[x_col].values.astype(float)
            y = d[y_col].values.astype(float)
            if x_min is None:
                x_min, x_max = x.min(), x.max()
            else:
                x_min = max(x_min, x.min())
                x_max = min(x_max, x.max())
            curves.append((x, y))

        if not curves or x_min is None or x_max is None or x_max <= x_min:
            continue

        # common grid
        grid = np.arange(x_min, x_max + 1e-9, grid_step, dtype=float)

        Ys = []
        for x, y in curves:
            # linear interpolation
            yi = np.interp(grid, x, y)
            Ys.append(yi)
        Y = np.stack(Ys, axis=0)
        mean = Y.mean(axis=0)
        std = Y.std(axis=0)

        plt.plot(grid, mean, label=f"{gid} (n={len(Ys)})")
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2)

    plt.xlabel(x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or metric)
    if len(groups) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True, help="Root directory containing many runs/")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to save plots and summaries")
    ap.add_argument("--no_group", action="store_true", help="Plot each run separately (no mean/std grouping)")
    ap.add_argument("--group_keys", nargs="*", default=["algo", "loss", "n_step", "arch"],
                    help="Config keys to define a group (for mean/std curves)")
    ap.add_argument("--threshold", type=float, default=0.8, help="Win rate threshold for steps-to-threshold")
    ap.add_argument("--k", type=int, default=3, help="Consecutive eval points required to count as reached")
    ap.add_argument("--grid_step", type=int, default=100, help="Interpolation step size for grouped plots")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_runs(run_root)
    if not run_dirs:
        print(f"[WARN] No runs found under: {run_root}")
        return

    # Load runs
    train_runs = []
    eval_runs = []
    threshold_rows = []

    for rd in run_dirs:
        cfg = safe_read_json(rd / "config.json")
        train_df = safe_read_csv(rd / "train.csv")
        eval_df = safe_read_csv(rd / "eval.csv")

        gid = group_id_from_cfg(cfg, args.group_keys) if not args.no_group else "ALL"
        label_train = get_label(rd, cfg, None if args.no_group else args.group_keys)
        label_eval = label_train

        train_runs.append({
            "run_dir": str(rd),
            "cfg": cfg,
            "group_id": gid,
            "label": label_train,
            "df": train_df,
        })
        eval_runs.append({
            "run_dir": str(rd),
            "cfg": cfg,
            "group_id": gid,
            "label": label_eval,
            "df": eval_df,
        })

        step_to_th, reached = compute_step_to_threshold(eval_df, args.threshold, args.k)
        threshold_rows.append({
            "run_dir": str(rd),
            "exp_name": cfg.get("exp_name", ""),
            "seed": cfg.get("seed", ""),
            "run_id": cfg.get("run_id", ""),
            "group_id": gid if not args.no_group else "",
            "threshold": args.threshold,
            "k": args.k,
            "step_to_threshold": step_to_th if step_to_th is not None else np.nan,
            "reached": int(reached),
        })

    # Save threshold summary
    thr_df = pd.DataFrame(threshold_rows)
    thr_df.to_csv(out_dir / "threshold_summary.csv", index=False)

    # Plot configs (y_col, title, filename)
    eval_plots = [
        ("win_rate", "Eval win_rate vs global_step", "eval_win_rate_vs_steps.png"),
        ("score_diff_mean", "Eval score_diff vs global_step", "eval_score_diff_vs_steps.png"),
    ]
    train_plots = [
        ("loss_mean", "Train loss_mean vs global_step", "train_loss_vs_steps.png"),
        ("td_error_p95", "Train td_error_p95 vs global_step", "train_td_error_p95_vs_steps.png"),
        ("q_max_mean", "Train q_max_mean vs global_step", "train_q_max_vs_steps.png"),
    ]

    if args.no_group:
        # each run is one line
        for y, title, fn in eval_plots:
            plot_raw_runs(
                [{"label": r["label"], "df": r["df"]} for r in eval_runs],
                out_dir, metric=y, x_col="global_step", y_col=y,
                title=title, ylabel=y, filename=fn
            )
        for y, title, fn in train_plots:
            plot_raw_runs(
                [{"label": r["label"], "df": r["df"]} for r in train_runs],
                out_dir, metric=y, x_col="global_step", y_col=y,
                title=title, ylabel=y, filename=fn
            )
    else:
        # group mean +/- std
        for y, title, fn in eval_plots:
            plot_grouped_runs(
                eval_runs, out_dir, metric=y,
                x_col="global_step", y_col=y,
                title=title, ylabel=y, filename=fn,
                grid_step=args.grid_step
            )
        for y, title, fn in train_plots:
            plot_grouped_runs(
                train_runs, out_dir, metric=y,
                x_col="global_step", y_col=y,
                title=title, ylabel=y, filename=fn,
                grid_step=args.grid_step
            )

    print(f"[OK] Saved plots + threshold_summary.csv to: {out_dir}")


if __name__ == "__main__":
    main()
