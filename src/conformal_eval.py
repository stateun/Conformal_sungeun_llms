# src/conformal_eval.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve

# --- robust imports: work with both `python src/...` and `python -m src...`
ROOT = Path(__file__).resolve().parent          # .../src
REPO = ROOT.parent                              # repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
try:
    from src.data_utils import load_data, DATA_MAP
except ImportError:
    from data_utils import load_data, DATA_MAP


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=[d.lower() for d in DATA_MAP.keys()])
    p.add_argument("--setting", type=str, default="semi_supervised",
                   choices=["semi_supervised", "unsupervised"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_idx", type=int, required=True)
    p.add_argument("--exp_dir", type=str, default=None)

    # 파일 선택/방향/파라미터
    p.add_argument("--test_score_glob", type=str, default="*_test.npy",
                   help="scores/ 아래에서 테스트 점수 파일 glob (기본: *_test.npy)")
    p.add_argument("--right_tail", action="store_true", default=True,
                   help="점수가 클수록 더 이상치이면 True (우꼬리).")
    p.add_argument("--fdr", type=float, default=0.1, help="BH 타깃 FDR q")
    p.add_argument("--delta", type=float, default=0.1, help="CCV 신뢰수준 (1-delta)")
    p.add_argument("--h", type=str, default="simes", choices=["simes"],
                   help="CCV 보정 함수 h 선택")
    p.add_argument("--only_pvalues", action="store_true",
                   help="marginal/CCV p-values만 계산·저장하고 종료(BH/FDR 스킵).")
    p.add_argument("--alpha_list", type=str, default="0.01,0.05,0.10",
                   help="콤마로 구분된 p-value 임계값 리스트 (예: '0.01,0.05,0.1').")
    # NEW: 테스트 라벨 기반 최대 F1 임계값(분석용) 자동 선택
    p.add_argument("--alpha_auto_f1", action="store_true",
                   help="라벨을 이용해 F1이 최대가 되는 p-value 임계값(alpha*)을 자동 선택(분석용).")
    return p.parse_args()


# ---- Marginal conformal p-values ----
def conformal_pvalues_marginal(cal_scores: np.ndarray, test_scores: np.ndarray, right_tail: bool = True):
    cal = np.asarray(cal_scores, dtype=float)
    tst = np.asarray(test_scores, dtype=float)
    n = cal.size
    cal_sorted = np.sort(cal)
    if right_tail:
        # p = (1 + # {cal >= s}) / (n + 1)
        ranks = np.searchsorted(cal_sorted, tst, side="right")  # <= s 개수
        num_ge = n - ranks
        pvals = (1.0 + num_ge) / (n + 1.0)
    else:
        # p = (1 + # {cal <= s}) / (n + 1)
        ranks = np.searchsorted(cal_sorted, tst, side="right")
        pvals = (1.0 + ranks) / (n + 1.0)
    return np.clip(pvals, 0.0, 1.0)


# ---- CCV: Simes 보정 (간단/실전용) ----
def simes_adjustment(u_marg: np.ndarray, n_cal: int, delta: float = 0.1, k: int | None = None):
    import math
    n = int(n_cal)
    if k is None:
        k = max(1, n // 2)
    # 경계함수 b(i) 구성 (간략 구현)
    log_c = math.log1p(-float(delta)) / k  # log(1-δ)/k
    bs = np.ones(n + 1, dtype=float)
    for i in range(1, n + 1):
        top = 0.0
        bot = 0.0
        valid = True
        for t in range(k):
            if i - t <= 0:
                valid = False
                break
            top += math.log(i - t)
            bot += math.log(n - t)
        if not valid:
            val = 1.0
        else:
            val = 1.0 - math.exp(log_c + (top - bot) / k)
            val = min(max(val, 0.0), 1.0)
        bs[n + 1 - i] = val
    bs[0] = 0.0
    bs[n] = 1.0

    idx = np.ceil((n + 1) * np.clip(u_marg, 0, 1)).astype(int)
    idx = np.clip(idx, 1, n)
    return bs[idx]


# ---- BH (FDR 제어) ----
def bh_binary_decisions(pvals: np.ndarray, q: float):
    m = len(pvals)
    order = np.argsort(pvals)
    pv_sort = pvals[order]
    thresh = q * (np.arange(1, m + 1) / m)
    ok = pv_sort <= thresh
    if not np.any(ok):
        return np.zeros(m, dtype=int), None
    k = int(np.max(np.where(ok)[0]))
    cutoff = float(pv_sort[k])
    y_pred = (pvals <= cutoff).astype(int)
    return y_pred, cutoff


# ---- helpers for adaptive alpha ---
def _confmat_metrics(y_true, yhat, eps=1e-12):
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + eps)
    tpr = tp / (tp + fn + eps)  # recall
    prec = tp / (tp + fp + eps)
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec + eps)
    return dict(FPR=fpr, TPR=tpr, Precision=prec, Recall=rec, F1=f1,
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

def best_alpha_by_f1(pvals, y_true):
    """모든 고유 p에 대해 yhat=(p<=alpha)로 F1 계산, 최대 F1의 alpha 반환.
       동률이면 더 작은 alpha(보수적) 선택."""
    p = np.asarray(pvals, dtype=float)
    uniq = np.unique(p)
    best = {"alpha": None, "F1": -1.0, "metrics": None}
    for a in uniq:
        yhat = (p <= a).astype(int)
        m = _confmat_metrics(y_true, yhat)
        if (m["F1"] > best["F1"] + 1e-12) or (
            abs(m["F1"] - best["F1"]) <= 1e-12 and (best["alpha"] is None or a < best["alpha"])
        ):
            best.update(alpha=float(a), F1=float(m["F1"]), metrics=m)
    return best


def load_scores_dir(args) -> Path:
    if args.exp_dir is None:
        exp_dir = Path("exp") / args.dataset / args.setting / f"split{args.n_splits}" / f"split{args.split_idx}"
    else:
        exp_dir = Path(args.exp_dir)
    scores_dir = exp_dir / "scores"
    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores dir not found: {scores_dir}")
    return scores_dir


def main():
    args = get_args()

    # 데이터 & 정답 (테스트 라벨)
    X_train, X_test, y_train, y_test = load_data(args)
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    scores_dir = load_scores_dir(args)

    # 파일 로드
    cal_path = scores_dir / "cal_scores_inlier.npy"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Missing calibration scores: {cal_path}\n"
            f"Run evaluate_anollm.py with --eval_split calibration first."
        )
    cal_scores = np.load(cal_path)

    # test score 파일 선택: glob 중 mtime 최신
    cand = list(scores_dir.glob(args.test_score_glob))
    if len(cand) == 0:
        raise FileNotFoundError(f"No test score files matching {args.test_score_glob} in {scores_dir}")
    test_path = max(cand, key=lambda p: p.stat().st_mtime)
    test_scores = np.load(test_path)

    if len(test_scores) != len(y_test):
        raise ValueError(f"Length mismatch: test_scores={len(test_scores)} vs y_test={len(y_test)}")

    # 1) marginal p-values
    p_marg = conformal_pvalues_marginal(cal_scores, test_scores, right_tail=args.right_tail)

    # 2) CCV (Simes)
    if args.h == "simes":
        p_ccv = simes_adjustment(p_marg, n_cal=len(cal_scores), delta=args.delta)
    else:
        raise NotImplementedError(args.h)

    # --- 저장 (p-values)
    np.save(scores_dir / "pvalues_marginal.npy", p_marg)
    np.save(scores_dir / f"pvalues_ccv_{args.h}_delta{args.delta}.npy", p_ccv)

    # --- only_pvalues: 요약 출력 후 종료 (BH/FDR 스킵) ---
    if args.only_pvalues:
        # ========= Threshold-free: AUC + "F1 최대" 운영점 지표 =========
        print("\n[Threshold-free AUC]")
        rows_auc = []

        def auc_and_best_op(name: str, score_vec):
            # AUC들
            auc_roc = roc_auc_score(y_test, score_vec)
            auc_pr  = average_precision_score(y_test, score_vec)

            # PR-curve로 F1 최대 운영점 찾기
            prec, rec, thr = precision_recall_curve(y_test, score_vec)
            f1_vals = 2 * prec * rec / (prec + rec + 1e-12)
            best_idx = int(np.nanargmax(f1_vals))

            # threshold 인덱스 매핑: prec/rec 길이는 len(thr)+1
            if best_idx == 0:
                t_best = np.inf  # 모두 음성으로 가는 임계(실제로는 거의 없음)
            else:
                t_best = float(thr[best_idx - 1])

            yhat = (score_vec >= t_best).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn + 1e-12)
            tpr = tp / (tp + fn + 1e-12)
            prec_best = tp / (tp + fp + 1e-12)
            rec_best = tpr
            f1_best = 2 * prec_best * rec_best / (prec_best + rec_best + 1e-12)
            pos_best = int(yhat.sum())

            rows_auc.append({
                "name": name, "AUC_ROC": auc_roc, "AUC_PR": auc_pr,
                "thr_bestF1": t_best, "pos_bestF1": pos_best,
                "FPR_bestF1": fpr, "TPR_bestF1": tpr,
                "Precision_bestF1": prec_best, "Recall_bestF1": rec_best, "F1_bestF1": f1_best
            })
            print(f"  {name:14s} | AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
                  f"[best-F1: thr={t_best:.6g}, pos={pos_best}, FPR={fpr:.4f}, TPR={tpr:.4f}, "
                  f"P={prec_best:.4f}, R={rec_best:.4f}, F1={f1_best:.4f}]")

        # 1) 원본 이상치 점수(클수록 이상): 그대로
        auc_and_best_op("raw_score", test_scores)
        # 2) marginal p-value (작을수록 이상) → 부호 반전해서 점수로
        auc_and_best_op("neg_p_marg", -p_marg)
        # 3) CCV(Simes) p-value (작을수록 이상) → 부호 반전
        auc_and_best_op("neg_p_ccv", -p_ccv)

        # AUC 요약 CSV 저장
        df_auc = pd.DataFrame(rows_auc)
        out_auc = scores_dir / "conformal_auc_summary.csv"
        df_auc.to_csv(out_auc, index=False)
        print(f"Saved AUC summary → {out_auc}")

        # ========= Metrics @ alpha: AUC 열을 함께 출력/저장 =========
        alphas = [float(x) for x in args.alpha_list.split(",")]
        rows = []
        eps = 1e-12

        # 메소드별 AUC(negated p) 미리 구해놓고 붙임
        auc_neg_p_marg_roc = roc_auc_score(y_test, -p_marg)
        auc_neg_p_marg_pr  = average_precision_score(y_test, -p_marg)
        auc_neg_p_ccv_roc  = roc_auc_score(y_test, -p_ccv)
        auc_neg_p_ccv_pr   = average_precision_score(y_test, -p_ccv)

        def metrics_at(p, alpha):
            yhat = (p <= alpha).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn + eps)
            tpr = tp / (tp + fn + eps)  # recall
            prec = tp / (tp + fp + eps)
            rec  = tpr
            f1   = 2 * prec * rec / (prec + rec + eps)
            return yhat.sum(), fpr, tpr, prec, rec, f1

        print("\n[Metrics @ alpha]")
        for name, p, auc_pair in [
            ("marginal", p_marg, (auc_neg_p_marg_roc, auc_neg_p_marg_pr)),
            (f"ccv_{args.h}", p_ccv, (auc_neg_p_ccv_roc,  auc_neg_p_ccv_pr)),
        ]:
            for alpha in alphas:
                cnt, fpr, tpr, prec, rec, f1 = metrics_at(p, alpha)
                rows.append({
                    "method": name, "alpha": alpha, "positives": int(cnt),
                    "FPR": fpr, "TPR": tpr, "Precision": prec, "Recall": rec, "F1": f1,
                    "AUC_ROC(neg_score)": auc_pair[0], "AUC_PR(neg_score)": auc_pair[1],
                })
                print(f"  {name:12s} @ α={alpha:>5.3f} | pos={cnt:4d} "
                      f"FPR={fpr:.4f} TPR={tpr:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
                      f"| AUC-ROC={auc_pair[0]:.4f} AUC-PR={auc_pair[1]:.4f}")

        # CSV 저장
        df = pd.DataFrame(rows)
        out_csv = scores_dir / "conformal_metrics_at_alpha.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved metrics @ alpha → {out_csv}")

        # ========= Adaptive alpha by max-F1 (분석용) =========
        if args.alpha_auto_f1:
            rows_auto = []

            best_marg = best_alpha_by_f1(p_marg, y_test)
            yhat_marg = (p_marg <= best_marg["alpha"]).astype(int)
            rows_auto.append({
                "method": "marginal",
                "alpha*:F1max": best_marg["alpha"],
                **best_marg["metrics"],
            })
            np.save(scores_dir / "decisions_marginal_alpha_f1.npy", yhat_marg)
            with open(scores_dir / "alpha_star_marginal_f1.txt", "w") as f:
                f.write(str(best_marg["alpha"]))

            best_ccv = best_alpha_by_f1(p_ccv, y_test)
            yhat_ccv = (p_ccv <= best_ccv["alpha"]).astype(int)
            rows_auto.append({
                "method": f"ccv_{args.h}",
                "alpha*:F1max": best_ccv["alpha"],
                **best_ccv["metrics"],
            })
            np.save(scores_dir / f"decisions_ccv_{args.h}_alpha_f1.npy", yhat_ccv)
            with open(scores_dir / f"alpha_star_ccv_{args.h}_f1.txt", "w") as f:
                f.write(str(best_ccv["alpha"]))

            # 콘솔 출력
            print("\n[Adaptive α by max-F1]")
            for r in rows_auto:
                print(f"  {r['method']:12s} | alpha*={r['alpha*:F1max']:.6g}  "
                      f"F1={r['F1']:.4f}  P={r['Precision']:.4f}  R={r['Recall']:.4f}  "
                      f"FPR={r['FPR']:.4f}  (TP={r['TP']}, FP={r['FP']}, TN={r['TN']}, FN={r['FN']})")

            # CSV 저장
            df_auto = pd.DataFrame(rows_auto)
            out_auto = scores_dir / "conformal_alpha_auto_f1_summary.csv"
            df_auto.to_csv(out_auto, index=False)
            print(f"Saved alpha* (F1-max) summary → {out_auto}")

        return  # only_pvalues 종료

    # --- (옵션) BH (FDR) 판정 및 요약 ---
    yhat_marg, cut_marg = bh_binary_decisions(p_marg, q=args.fdr)
    yhat_ccv,  cut_ccv  = bh_binary_decisions(p_ccv,  q=args.fdr)

    np.save(scores_dir / f"decisions_bh_marginal_fdr{args.fdr}.npy", yhat_marg)
    np.save(scores_dir / f"decisions_bh_ccv_{args.h}_fdr{args.fdr}_delta{args.delta}.npy", yhat_ccv)

    def summary(name, yhat, cut):
        tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-12)
        tpr = tp / (tp + fn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        rec  = tpr
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        print(f"[{name}] count={int(yhat.sum())}/{len(yhat)}  cutoff={cut if cut is not None else float('nan'):.6g}  "
              f"FPR={fpr:.4f}  TPR={tpr:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    print(f"\n== Conformal on {args.dataset} split{args.split_idx} "
          f"(cal={len(cal_scores)}, test={len(test_scores)}), FDR q={args.fdr}, delta={args.delta} ==")
    print(f"AUC-ROC(raw scores)={roc_auc_score(y_test, test_scores):.4f}  "
          f"AUC-PR(raw scores)={average_precision_score(y_test, test_scores):.4f}")
    summary("Marginal + BH", yhat_marg, cut_marg)
    summary(f"CCV-{args.h} + BH", yhat_ccv, cut_ccv)


if __name__ == "__main__":
    main()
