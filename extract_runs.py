import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple


REQUIRED_FILES = ["config.json", "train.csv", "eval.csv"]


def find_runs(run_root: Path) -> List[Path]:
    """Recursively find run directories that contain REQUIRED_FILES."""
    run_dirs: List[Path] = []
    for cfg_path in run_root.rglob("config.json"):
        run_dir = cfg_path.parent
        ok = True
        for fn in REQUIRED_FILES:
            if not (run_dir / fn).exists():
                ok = False
                break
        if ok:
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def parse_step_from_checkpoint(name: str) -> Optional[int]:
    """
    Supported patterns:
    - eval_checkpoint_ep{ep}_step{step}.pth
    - eval_checkpoint_{step}.pth (legacy)
    """
    if not name.startswith("eval_checkpoint"):
        return None
    m = re.search(r"_step(\d+)\.pth$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^eval_checkpoint_(\d+)\.pth$", name)
    if m:
        return int(m.group(2))
    return None


def parse_step_from_video(name: str) -> Optional[int]:
    m = re.search(r"_step_(\d+)\.mp4$", name)
    if m:
        return int(m.group(1))
    return None


def pick_latest_by_step(
    files: Iterable[Path],
    step_fn: Callable[[str], Optional[int]],
) -> Tuple[Optional[int], List[Path]]:
    best_step: Optional[int] = None
    best_files: List[Path] = []
    for p in files:
        step = step_fn(p.name)
        if step is None:
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_files = [p]
        elif step == best_step:
            best_files.append(p)
    return best_step, best_files


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] copytree {src} -> {dst}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_run(run_dir: Path, out_root: Path, run_root: Path, dry_run: bool) -> None:
    rel = run_dir.relative_to(run_root)
    out_run = out_root / rel
    out_run.mkdir(parents=True, exist_ok=True)

    # Copy top-level files in the run dir.
    for p in run_dir.iterdir():
        if p.is_file():
            copy_file(p, out_run / p.name, dry_run)

    # Handle checkpoints and videos with special logic.
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        ckpt_files = [p for p in checkpoints_dir.iterdir() if p.is_file()]
        _, latest_ckpts = pick_latest_by_step(ckpt_files, parse_step_from_checkpoint)
        for p in latest_ckpts:
            copy_file(p, out_run / "checkpoints" / p.name, dry_run)

    videos_dir = run_dir / "videos"
    if videos_dir.is_dir():
        video_files = [p for p in videos_dir.iterdir() if p.is_file()]
        _, latest_videos = pick_latest_by_step(video_files, parse_step_from_video)
        for p in latest_videos:
            copy_file(p, out_run / "videos" / p.name, dry_run)

    # Copy any other subdirs as-is (except checkpoints/videos).
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name in ("checkpoints", "videos"):
            continue
        copy_tree(p, out_run / p.name, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True, help="Root directory containing runs/")
    ap.add_argument("--out_root", type=str, required=True, help="Where to write the extracted copy")
    ap.add_argument("--dry_run", action="store_true", help="Print actions without copying files")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = find_runs(run_root)
    if not run_dirs:
        print(f"[WARN] No runs found under: {run_root}")
        return

    for rd in run_dirs:
        extract_run(rd, out_root, run_root, args.dry_run)

    print(f"[OK] Extracted {len(run_dirs)} runs into: {out_root}")


if __name__ == "__main__":
    main()
