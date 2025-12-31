import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, Optional


class RunLogger:
    """
    Minimal run logger for training/evaluation metrics.

    Creates a run directory, writes config.json, and appends to train.csv / eval.csv
    with stable headers so downstream aggregation stays consistent as features grow.
    """

    TRAIN_COLUMNS = [
        "run_id",
        "seed",
        "episode",
        "global_step",
        "episode_len",
        "train_return",
        "train_score_diff",
        "epsilon",
        "loss_mean",
        "td_error_p95",
        "q_max_mean",
    ]

    EVAL_COLUMNS = [
        "run_id",
        "seed",
        "eval_idx",
        "global_step",
        "eval_episodes",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "score_diff_mean",
        "score_diff_std",
        "return_mean",
        "return_std",
        "avg_episode_len",
    ]

    def __init__(self, run_dir: str, run_id: str, seed: int):
        self.run_dir = run_dir
        self.run_id = run_id
        self.seed = seed

        os.makedirs(self.run_dir, exist_ok=True)

        self._train_fp = None
        self._train_writer: Optional[csv.DictWriter] = None
        self._eval_fp = None
        self._eval_writer: Optional[csv.DictWriter] = None

    @staticmethod
    def default_run_id(exp_name: str, seed: int) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{exp_name}_seed{seed}_{ts}"

    def _ensure_writer(
        self, file_path: str, columns: Iterable[str], kind: str
    ) -> csv.DictWriter:
        file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        fp = open(file_path, mode="a", newline="")
        writer = csv.DictWriter(fp, fieldnames=list(columns))
        if not file_exists:
            writer.writeheader()
        if kind == "train":
            self._train_fp = fp
            self._train_writer = writer
        else:
            self._eval_fp = fp
            self._eval_writer = writer
        return writer

    def _write_row(self, writer: csv.DictWriter, row: Dict, columns: Iterable[str], fp):
        safe_row = {}
        for col in columns:
            val = row.get(col)
            if val is None:
                val = float("nan")
            safe_row[col] = val
        writer.writerow(safe_row)
        fp.flush()

    def log_config(self, config_dict: Dict):
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def log_train_episode(self, row_dict: Dict):
        if self._train_writer is None:
            train_path = os.path.join(self.run_dir, "train.csv")
            self._ensure_writer(train_path, self.TRAIN_COLUMNS, kind="train")
        self._write_row(self._train_writer, row_dict, self.TRAIN_COLUMNS, self._train_fp)

    def log_eval(self, row_dict: Dict):
        if self._eval_writer is None:
            eval_path = os.path.join(self.run_dir, "eval.csv")
            self._ensure_writer(eval_path, self.EVAL_COLUMNS, kind="eval")
        self._write_row(self._eval_writer, row_dict, self.EVAL_COLUMNS, self._eval_fp)

    def close(self):
        if self._train_fp:
            self._train_fp.close()
        if self._eval_fp:
            self._eval_fp.close()
