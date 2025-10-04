import os
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from anollm import AnoLLM
from src.data_utils import load_data, DATA_MAP, get_text_columns, get_max_length_dict
from train_anollm import get_run_name


# -------------------------
# Args
# -------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wine",
                        choices=[d.lower() for d in DATA_MAP.keys()],
                        help="Name of datasets in the ODDS benchmark")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--setting", type=str, default="semi_supervised",
                        choices=["semi_supervised", "unsupervised"],
                        help="semi_supervised:an uncontaminated, unsupervised setting; unsupervised:a contaminated, unsupervised setting")

    # dataset hyperparameters
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_idx", type=int, default=None)  # 0..n_splits-1

    # binning
    parser.add_argument("--binning", type=str,
                        choices=["quantile", "equal_width", "language", "none", "standard"],
                        default="standard")
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--remove_feature_name", action="store_true")

    # model hyperparameters (for getting the model name)
    parser.add_argument("--model", type=str,
                        choices=["gpt2", "distilgpt2", "smol", "smol-360", "smol-1.7b"],
                        default="smol")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--no_random_permutation", action="store_true", default=False)

    # scoring
    parser.add_argument("--batch_size", type=int, default=128)      # per GPU
    parser.add_argument("--n_permutations", type=int, default=100)  # total; will be split across ranks

    # which split to score
    parser.add_argument("--eval_split", type=str, choices=["test", "calibration"],
                        default="test",
                        help="Which split to score and save: test or calibration (normals only).")
    parser.add_argument("--cal_ratio", type=float, default=0.2,
                        help="Fraction of TRAIN NORMALS used for calibration (when --eval_split=calibration).")

    args = parser.parse_args()

    # map shorthand -> HF repo
    if args.model == "smol":
        args.model = "HuggingFaceTB/SmolLM-135M"
    elif args.model == "smol-360":
        args.model = "HuggingFaceTB/SmolLM-360M"
    elif args.model == "smol-1.7b":
        args.model = "HuggingFaceTB/SmolLM-1.7B"

    return args


# -------------------------
# Main
# -------------------------
def main():
    # DDP world
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    args = get_args()

    # experiment dir
    if args.exp_dir is None:
        args.exp_dir = (
            Path("exp")
            / args.dataset
            / args.setting
            / f"split{args.n_splits}"
            / f"split{args.split_idx}"
        )
    if not os.path.exists(args.exp_dir):
        raise ValueError(f"Experiment directory {args.exp_dir} does not exist")

    score_dir = Path(args.exp_dir) / "scores"
    run_name = get_run_name(args)  # e.g. anollm_lr5e-05_standard_smolLM_test

    # final output path (mean across permutations)
    if args.eval_split == "calibration":
        out_path = score_dir / "cal_scores_inlier.npy"
        runtime_subdir = "calibration"
    else:
        out_path = score_dir / f"{run_name}.npy"
        runtime_subdir = "test"

    if dist.get_rank() == 0:
        score_dir.mkdir(parents=True, exist_ok=True)

    # load data
    X_train, X_test, y_train, y_test = load_data(args)

    # pick eval split
    if args.eval_split == "calibration":
        ytr = np.asarray(y_train)
        idx_norm = np.where(ytr == 0)[0]
        if len(idx_norm) == 0:
            raise ValueError("No normal (label=0) samples in TRAIN to build calibration set.")
        rng = np.random.default_rng(getattr(args, "seed", 42))
        rng.shuffle(idx_norm)
        n_cal = max(1, int(len(idx_norm) * args.cal_ratio))
        cal_idx = idx_norm[:n_cal]
        X_eval = X_train.iloc[cal_idx] if isinstance(X_train, pd.DataFrame) else X_train[cal_idx]
        # y_eval = ytr[cal_idx]  # not used, but can be kept if needed
    else:
        X_eval = X_test
        # y_eval = y_test

    # build model wrapper
    model_dir = Path(args.exp_dir) / "models"
    model_path = model_dir / f"{run_name}.pt"

    efficient_finetuning = "lora" if args.lora else ""
    max_length_dict = get_max_length_dict(args.dataset)
    text_columns = get_text_columns(args.dataset)

    model = AnoLLM(
        args.model,
        efficient_finetuning=efficient_finetuning,
        model_path=model_path,
        max_length_dict=max_length_dict,
        textual_columns=text_columns,
        no_random_permutation=args.no_random_permutation,
        bp16=True,
    )
    print(text_columns, max_length_dict)

    # load weights
    model.load_from_state_dict(model_path)
    model.model.to(local_rank)

    # wrap with DDP (even for nproc=1 this is safe)
    model.model = torch.nn.parallel.DistributedDataParallel(
        model.model, device_ids=[local_rank], output_device=local_rank
    )

    # split permutations across ranks
    remainder = args.n_permutations % world_size
    n_perm = int(args.n_permutations / world_size)
    if local_rank < remainder:
        n_perm += 1

    # ---- scoring ----
    start_time = time.time()
    model.model.eval()
    with torch.no_grad():
        scores_part = model.decision_function(
            X_eval,
            n_permutations=n_perm,
            batch_size=args.batch_size,
            device=device,
        )
    end_time = time.time()

    # gather all ranks
    all_parts = [None for _ in range(world_size)]
    dist.all_gather_object(all_parts, scores_part)

    if dist.get_rank() == 0:
        # numpy 배열로 강제 변환 + 불필요한 차원 제거
        all_parts = [np.asarray(p) for p in all_parts]
        all_parts = [p.squeeze() for p in all_parts]

        if all_parts[0].ndim == 2:
            # 각 rank가 (N, n_perm_part) 반환 → 열 방향으로 이어붙인 뒤 평균
            all_scores = np.concatenate(all_parts, axis=1)        # (N, n_perm_total)
        elif all_parts[0].ndim == 1:
            # 각 rank가 (N,) 반환 → 열로 쌓기
            all_scores = np.stack(all_parts, axis=1)              # (N, world_size)
        else:
            raise ValueError(f"Unexpected score shape: {all_parts[0].shape}")

        mean_scores = all_scores.mean(axis=1).astype(np.float32)

        score_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mean_scores)
        raw_path = score_dir / (f"raw_{run_name}.npy" if args.eval_split == "test"
                                else "raw_cal_scores_inlier.npy")
        np.save(raw_path, all_scores)

        print(f"[evaluate_anollm] saved scores: {out_path}  shape={mean_scores.shape}")
        print(f"[evaluate_anollm] saved raw:    {raw_path}  shape={all_scores.shape}")

        print(f"Inference time: {end_time - start_time:.3f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
