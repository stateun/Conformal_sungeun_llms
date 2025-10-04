import os
from pathlib import Path
import argparse

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import pandas as pd
from data_utils import load_data, DATA_MAP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='wine',
                        choices=[d.lower() for d in DATA_MAP.keys()],
                        help="Name of datasets in the ODDS benchmark")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--setting", type=str, default='semi_supervised',
                        choices=['semi_supervised', 'unsupervised'])

    # dataset hyperparameters
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_idx", type=int, default=None)  # 0 to n_splits-1

    args = parser.parse_args()
    return args


def metrics_at_f1_max_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    Find the threshold (tau*) where F1 is maximized on (y_true, y_score),
    and return P/R/F1 plus FPR/TPR at that threshold.
    """
    # precision_recall_curve returns precision/recall arrays with length = len(thresholds)+1
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Edge case: thresholds may be empty if y_true has a single class
    if thresholds.size == 0:
        # Fallback: no thresholding possible; return NaNs
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # thresholds align with precision[1:], recall[1:]
    f1_arr = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-12)
    best_idx = int(np.argmax(f1_arr))
    tau_star = thresholds[best_idx]

    # Predictions at tau*
    y_pred = (y_score >= tau_star).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)

    P = precision[best_idx + 1]
    R = recall[best_idx + 1]
    F1 = f1_arr[best_idx]

    return tau_star, P, R, F1, fpr, tpr


def tabular_metrics(y_true, y_score, seed: int = 42):
    """
    Calculates evaluation metrics for tabular anomaly detection.

    Returns:
        tuple:
          - auc_roc
          - auc_pr
          - f1_at_k
          - p_at_k
          - r_at_k
          - f1_at_tau (F1 at the threshold where F1 is maximized)
          - p_at_tau
          - r_at_tau
          - fpr_at_tau
          - tpr_at_tau
          - tau_star (threshold achieving max F1)
    """
    # --- Shuffle to mitigate ties bias for argpartition(F1@K) ---
    n_test = len(y_true)
    rng = np.random.default_rng(seed)
    new_index = rng.permutation(n_test)
    y_true = np.asarray(y_true)[new_index]
    y_score = np.asarray(y_score)[new_index]

    # ----- AUCs (threshold-free) -----
    auc_roc = metrics.roc_auc_score(y_true, y_score)
    auc_pr = metrics.average_precision_score(y_true, y_score)

    # ----- F1@K (K = #positives) -----
    top_k = int((y_true == 1).sum())
    indices = np.argpartition(y_score, -top_k)[-top_k:]
    y_pred_k = np.zeros_like(y_true, dtype=int)
    y_pred_k[indices] = 1

    p_at_k, r_at_k, f1_at_k, _ = metrics.precision_recall_fscore_support(
        y_true.astype(int), y_pred_k, average='binary', zero_division=0
    )

    # ----- F1-max threshold (tau*) -----
    tau_star, p_at_tau, r_at_tau, f1_at_tau, fpr_at_tau, tpr_at_tau = metrics_at_f1_max_threshold(
        y_true.astype(int), y_score
    )

    return (auc_roc, auc_pr,
            f1_at_k, p_at_k, r_at_k,
            f1_at_tau, p_at_tau, r_at_tau, fpr_at_tau, tpr_at_tau, tau_star)


def is_baseline(s: str):
    return False if 'anollm' in s else True


def get_metrics(args, only_raw=False, only_normalized=False, only_ordinal=False):
    X_train, X_test, y_train, y_test = load_data(args)
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    if args.exp_dir is None:
        args.exp_dir = Path('exp') / args.dataset / args.setting / \
                       f"split{args.n_splits}" / f"split{args.split_idx}"

    score_dir = args.exp_dir / 'scores'
    if not os.path.exists(score_dir):
        raise ValueError(f"Score directory {score_dir} does not exist")

    method_dict = {}
    for score_npy in os.listdir(score_dir):
        if not score_npy.endswith('.npy'):
            continue
        if score_npy.startswith('raw'):
            continue

        # baseline filters
        if is_baseline(score_npy) and only_normalized and ('normalized' not in score_npy):
            continue
        if is_baseline(score_npy) and only_ordinal and ('ordinal' not in score_npy):
            continue

        method = '.'.join(score_npy.split('.')[:-1])
        if method == 'rdp':
            continue

        scores = np.load(score_dir / score_npy)

        if np.isnan(scores).any():
            print(f"NaNs in scores for {method}")
            method_dict[method] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan]
            continue
        if np.isinf(scores).any():
            print(f"Infs in scores for {method}")
            method_dict[method] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan]
            continue

        (auc_roc, auc_pr,
         f1_k, p_k, r_k,
         f1_tau, p_tau, r_tau, fpr_tau, tpr_tau, tau_star) = tabular_metrics(y_test, scores)

        # NOTE: keep list form for compatibility with existing ranking code
        method_dict[method] = [auc_roc, auc_pr, f1_k, p_k, r_k,
                               f1_tau, p_tau, r_tau, fpr_tau, tpr_tau, tau_star]

    # rankings for the first 5 metrics (higher is better)
    rankings = []
    first_key = list(method_dict.keys())[0]
    num_metrics = len(method_dict[first_key])  # now 11
    for i in range(num_metrics):
        # By default: higher is better. We'll compute ranks for all anyway,
        # but we'll only DISPLAY ranks for the first 5 classic metrics below.
        scores_for_metric_i = [-method_dict[k][i] for k in method_dict.keys()]
        ranking = np.argsort(scores_for_metric_i).argsort() + 1
        rankings.append(ranking)

    print("-" * 100)
    for idx, (k, v) in enumerate(method_dict.items()):
        # Classic metrics (with ranks)
        line = (f"{k:30s}: "
                f"AUC-ROC: {v[0]:.4f} ({rankings[0][idx]:2d}), "
                f"AUC-PR: {v[1]:.4f} ({rankings[1][idx]:2d}), "
                f"F1@K: {v[2]:.4f} ({rankings[2][idx]:2d}), "
                f"P@K: {v[3]:.4f} ({rankings[3][idx]:2d}), "
                f"R@K: {v[4]:.4f} ({rankings[4][idx]:2d})")
        print(line)
        # Extras at F1-max threshold (no ranks shown; FPR lower is better)
        print(f"{'':30s}  "
              f"F1@τ*: {v[5]:.4f}, P@τ*: {v[6]:.4f}, R@τ*: {v[7]:.4f}, "
              f"FPR@τ*: {v[8]:.4f}, TPR@τ*: {v[9]:.4f}, thr: {v[10]:.6f}")

    return method_dict


def filter_results(d: dict):
    d2 = {}
    for k in d.keys():
        new_key = k
        if is_baseline(k):
            d2[k] = d[k]
        else:
            if '_lora' in k:
                temp = k.replace('_lora', '')
                if temp in d:
                    continue
                else:
                    new_key = new_key.replace('_lora', '')
                    d2[new_key] = d[k]
            else:
                d2[new_key] = d[k]
    return d2


def aggregate_results(m_dicts):
    # aggregate over splits
    keys = list(m_dicts[0].keys())
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1@K', 'P@K', 'R@K',
                    'F1@τ*', 'P@τ*', 'R@τ*', 'FPR@τ*', 'TPR@τ*', 'thr@τ*']
    aggregate = {k: {mn: [] for mn in metric_names} for k in keys}

    for i in range(len(m_dicts)):
        for k in keys:
            try:
                vals = m_dicts[i][k]
                aggregate[k]['AUC-ROC'].append(vals[0])
                aggregate[k]['AUC-PR'].append(vals[1])
                aggregate[k]['F1@K'].append(vals[2])
                aggregate[k]['P@K'].append(vals[3])
                aggregate[k]['R@K'].append(vals[4])
                aggregate[k]['F1@τ*'].append(vals[5])
                aggregate[k]['P@τ*'].append(vals[6])
                aggregate[k]['R@τ*'].append(vals[7])
                aggregate[k]['FPR@τ*'].append(vals[8])
                aggregate[k]['TPR@τ*'].append(vals[9])
                aggregate[k]['thr@τ*'].append(vals[10])
            except Exception as e:
                print("Incomplete results for ", k, "err:", e)
                if k in aggregate:
                    del aggregate[k]
                for j in range(len(m_dicts)):
                    if k in m_dicts[j]:
                        del m_dicts[j][k]
                continue

    print("-" * 100)

    # rankings for classic metrics only (higher is better)
    rankings = {}
    if aggregate:
        ref_key = list(aggregate.keys())[0]
        for metric_name in ['AUC-ROC', 'AUC-PR', 'F1@K', 'P@K', 'R@K']:
            scores = [-np.mean(aggregate[k][metric_name]) for k in aggregate.keys()]
            ranking = np.argsort(scores).argsort() + 1
            rankings[metric_name] = ranking

    for idx, k in enumerate(aggregate.keys()):
        msg = (f"{k:30s}: "
               f"AUC-ROC: {np.mean(aggregate[k]['AUC-ROC']):.4f} +- {np.std(aggregate[k]['AUC-ROC']):.4f} "
               f"({rankings.get('AUC-ROC', []) and rankings['AUC-ROC'][idx]:2d}), "
               f"AUC-PR: {np.mean(aggregate[k]['AUC-PR']):.4f} +- {np.std(aggregate[k]['AUC-PR']):.4f} "
               f"({rankings.get('AUC-PR', []) and rankings['AUC-PR'][idx]:2d}), "
               f"F1@K: {np.mean(aggregate[k]['F1@K']):.4f} +- {np.std(aggregate[k]['F1@K']):.4f} "
               f"({rankings.get('F1@K', []) and rankings['F1@K'][idx]:2d})  "
               f"P@K: {np.mean(aggregate[k]['P@K']):.4f} +- {np.std(aggregate[k]['P@K']):.4f} "
               f"({rankings.get('P@K', []) and rankings['P@K'][idx]:2d})  "
               f"R@K: {np.mean(aggregate[k]['R@K']):.4f} +- {np.std(aggregate[k]['R@K']):.4f} "
               f"({rankings.get('R@K', []) and rankings['R@K'][idx]:2d})")
        print(msg)
        # extras (no ranks)
        print(f"{'':30s}  "
              f"F1@τ*: {np.mean(aggregate[k]['F1@τ*']):.4f} +- {np.std(aggregate[k]['F1@τ*']):.4f}, "
              f"P@τ*: {np.mean(aggregate[k]['P@τ*']):.4f} +- {np.std(aggregate[k]['P@τ*']):.4f}, "
              f"R@τ*: {np.mean(aggregate[k]['R@τ*']):.4f} +- {np.std(aggregate[k]['R@τ*']):.4f}, "
              f"FPR@τ*: {np.mean(aggregate[k]['FPR@τ*']):.4f} +- {np.std(aggregate[k]['FPR@τ*']):.4f}, "
              f"TPR@τ*: {np.mean(aggregate[k]['TPR@τ*']):.4f} +- {np.std(aggregate[k]['TPR@τ*']):.4f}, "
              f"thr@τ*: {np.mean(aggregate[k]['thr@τ*']):.6f} +- {np.std(aggregate[k]['thr@τ*']):.6f}")
    return aggregate, rankings


def get_metrics_or_aggregate(args):
    if args.split_idx is None:
        L = []
        for i in range(args.n_splits):
            args.split_idx = i
            args.exp_dir = None
            results = get_metrics(args)
            L.append(results)
        aggregate_results(L)
    else:
        print(args)
        _ = get_metrics(args)


def main():
    args = get_args()
    get_metrics_or_aggregate(args)


if __name__ == '__main__':
    main()
