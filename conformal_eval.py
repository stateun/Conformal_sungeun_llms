# src/conformal_eval.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data_utils import load_data, DATA_MAP
from train_anollm import get_run_name


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=[d.lower() for d in DATA_MAP.keys()], required=True)
    p.add_argument("--setting", type=str, choices=["semi_supervised", "unsupervised"], default="semi_supervised")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_idx", type=int, required=True)
    p.add_argument("--binning", type=str, choices=["quantile", "equal_width", "language", "none", "standard"],
                   default="standard")
    p.add_argument("--n_buckets", type=int, default=10)
    p.add_argument("--remove_feature_name", action="store_true")

    # 모델 tag 추론을 위해 동일 인자 필요
    p.add_argument("--model", type=str, choices=["gpt2", "distilgpt2", "smol", "smol-360", "smol-1.7b"], default="smol")
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lr", type=float, default=5e-5)

    # conformal 옵션
    p.add_argument("--right_tail", action="store_true", default=True,
                   help="점수가 클수록 이상이면 True(기본), 반대면 False")
    p.add_argument("--h", type=str, choices=["simes"], default="simes",
                   help="CCV 보정 방식(현재 simes만 지원)")
    p.add_argument("--delta", type=float, default=0.10, help="CCV에서 사용할 델타(표기용)")
    p.add_argument("--only_pvalues", action="store_true",
                   help="p-value만 저장하고 종료(BH/FDR 등 판정 스킵)")
    return p.parse_args()


def map_model_shorthand(m: str) -> str:
    """train/evaluate에서 쓰던 모델 명명과 일치시키기 위해 필요 없음(파일명 생성에 영향 X).
    여기선 run_name은 train_anollm.get_run_name(args)로 통일합니다.
    """
    return m


def simes_adjustment(pvals: np.ndarray) -> np.ndarray:
    """Simes 보정: 정렬된 p(1..m)에 대해 adj_i = min_{k>=rank(i)} (m/k) * p_(k), 역정렬 복원."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    k = np.arange(1, n + 1, dtype=float)
    adj_sorted = (n / k) * p_sorted
    # 뒤에서부터 누적 최소로 단조성 보장
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    # [0,1]로 클리핑 후 원래 순서로 복원
    adj = np.empty_like(adj_sorted)
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    return adj


def marginal_pvalues(cal: np.ndarray, test: np.ndarray, right_tail: bool = True) -> np.ndarray:
    """Conformal marginal p-values (smoothed): (1 + #cal >= s_i)/(M+1) (우측 꼬리 기준)"""
    cal = np.asarray(cal, dtype=float)
    test = np.asarray(test, dtype=float)
    M = cal.size
    if right_tail:
        cnt = (cal[:, None] >= test[None, :]).sum(axis=0)
    else:
        cnt = (cal[:, None] <= test[None, :]).sum(axis=0)
    p = (1.0 + cnt) / (M + 1.0)
    return p


def main():
    args = get_args()

    # 실험 디렉토리
    exp_dir = Path("exp") / args.dataset / args.setting / f"split{args.n_splits}" / f"split{args.split_idx}"
    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # run_name으로 test 점수 파일 찾기
    run_name = get_run_name(args)  # 예: anollm_lr5e-05_standard_smolLM_test
    test_path = scores_dir / f"{run_name}.npy"
    cal_path = scores_dir / "cal_scores_inlier.npy"

    if not test_path.exists():
        raise FileNotFoundError(f"test scores not found: {test_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"calibration scores not found: {cal_path}")

    test_scores = np.load(test_path)          # (N,)
    cal_scores = np.load(cal_path)            # (M,)

    # p-values
    p_marg = marginal_pvalues(cal_scores, test_scores, right_tail=args.right_tail)
    np.save(scores_dir / "pvalues_marginal.npy", p_marg)

    if args.h == "simes":
        p_ccv = simes_adjustment(p_marg)
        np.save(scores_dir / f"pvalues_ccv_simes_delta{args.delta}.npy", p_ccv)
    else:
        raise NotImplementedError(args.h)

    # 요약 + (선택) AUC 출력
    print(f"[conformal_eval] dataset={args.dataset} split={args.split_idx}")
    print(f"  p<=0.01  marg={np.mean(p_marg<=0.01):.3f}  ccv={np.mean(p_ccv<=0.01):.3f}")
    print(f"  p<=0.05  marg={np.mean(p_marg<=0.05):.3f}  ccv={np.mean(p_ccv<=0.05):.3f}")
    print(f"  p<=0.10  marg={np.mean(p_marg<=0.10):.3f}  ccv={np.mean(p_ccv<=0.10):.3f}")

    if args.only_pvalues:
        # 참고용으로 원 점수의 AUC도 같이
        Xtr, Xte, ytr, yte = load_data(args)
        try:
            print(f"  AUC-ROC={roc_auc_score(yte, test_scores):.4f}  AUC-PR={average_precision_score(yte, test_scores):.4f}")
        except Exception as e:
            print(f"  AUC compute skipped: {e}")
        return

    # (여기 아래는 BH 등 추가 판정을 하고 싶을 때만 확장)
    # ...


if __name__ == "__main__":
    main()
