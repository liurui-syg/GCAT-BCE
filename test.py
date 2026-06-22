import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (precision_recall_curve, confusion_matrix, roc_auc_score, roc_curve, auc, recall_score,
                             precision_score, f1_score, matthews_corrcoef)
import pickle
from GCAT_BCE import GCN_GAT
import numpy as np
import random
import os
import time
import pandas as pd

torch.cuda.empty_cache()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_test(labels, outputs, threshold=0.18):
    outputs = torch.sigmoid(torch.from_numpy(outputs)).numpy()
    predicted = (outputs > threshold).astype(float)
    precision, recall, _ = precision_recall_curve(labels, outputs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()

    fpr, tpr, _ = roc_curve(labels, outputs)
    max_fpr = 0.1
    mask = fpr <= max_fpr
    fpr_filtered = np.append(fpr[mask], max_fpr)
    tpr_filtered = np.append(tpr[mask], np.interp(max_fpr, fpr, tpr))
    auc10 = auc(fpr_filtered, tpr_filtered) / max_fpr

    # --- AUC-PR10% ---
    sorted_indices = np.argsort(outputs)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]

    top10_count = max(1, int(len(sorted_labels) * 0.1))
    top10_labels = sorted_labels[:top10_count]
    top10_outputs = np.array(outputs)[sorted_indices[:top10_count]]

    if len(np.unique(top10_labels)) == 2:
        pr10_prec, pr10_recall, _ = precision_recall_curve(top10_labels, top10_outputs)
        auc_pr10 = auc(pr10_recall, pr10_prec)
    else:
        auc_pr10 = 0.0

    sensitivity = recall_score(labels, predicted)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    bac = (sensitivity + specificity) / 2

    return {
        'threshold': threshold,
        'accuracy': (predicted == labels).mean(),
        'auc_roc': roc_auc_score(labels, outputs),
        'auc_pr': auc(recall, precision),
        'mcc': matthews_corrcoef(labels, predicted),
        'recall': recall_score(labels, predicted),
        'specificity': tn / (tn + fp) if (tn + fp) != 0 else 0,
        'precision': precision_score(labels, predicted),
        'f1': f1_score(labels, predicted),
        'bac': round(bac, 4),
        'auc10%': round(auc10, 4),
        'auc_pr10%': round(auc_pr10, 4)
    }


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    sample_info = []
    first_flag = True

    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True)
            out = model(data.x, data.dist, data.adj)
            ys.append(data.y.cpu())
            preds.append(out.cpu())

            if first_flag:
                print("\n===== Data 对象包含的所有字段 =====")
                print(f"data keys: {data.keys()}")
                first_flag = False

            # PDB ID 使用 data.name
            pdb_name = data.name[0] if isinstance(data.name, list) else data.name
            node_num = data.y.size(0)

            for res_idx in range(node_num):
                pred_val = out[res_idx].cpu().item()
                label_val = data.y[res_idx].cpu().item()

                sample_info.append({
                        "pdb_id": pdb_name,
                        "residue_idx": res_idx,
                        "prediction": pred_val,
                        "label": label_val
                })

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    return y_true, y_pred, sample_info


def evaluate_with_threshold(y_true, y_pred, threshold):
    """单独封装：给定阈值，计算指标"""
    return compute_metrics_test(y_true, y_pred, threshold)


def test():
    start_time = time.time()

    # with open('feature/BP3_15_ordc+weighted_rsa0.15_euclid12.pkl', 'rb') as f:
    # with open('feature/BP3_582_ordc+laplacian_weighted_rsa0.15.pkl', 'rb') as f:
    # with open('feature_epitope3D_245/epitope3D_45_ordc+weighted_rsa0_euclid12.pkl', 'rb') as f:
    # with open("features_PDB2526_28(pdb)/PDB2526_28_ordc+weighted_rsa0.15_euclid12.pkl", 'rb') as f:
    # with open("features_RoBep_187(pdb)/RoBep_187_ordc+weighted_rsa0_euclid12.pkl", 'rb') as f:
    with open("features_RoBep_187(pdb)/RoBep_187_ordc+weighted_rsa0.15_euclid12.pkl", 'rb') as f:
        test_data = pickle.load(f)
    print("test_pkl = RoBep_187_ordc+weighted_rsa0.15_euclid12.pkl")

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Use device: {device}")
    model = GCN_GAT().to(device)
    print(f"Model device: {next(model.parameters()).device}")

    best_model_path = "best_model_GCAT-BCE.pt"
    print("best_model = best_model_GCAT-BCE.pt")

    print("\n[Testing best model on test set]")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    y_true, y_pred, sample_info = evaluate(model, test_loader, device)

    result_df = pd.DataFrame(sample_info)
    csv_save_path = "GCAT-BCE_RoBep_187_results.csv"
    result_df.to_csv(csv_save_path, index=False, encoding='utf-8')
    print(f"\n预测结果已保存至: {csv_save_path}")

    print("\n" + "=" * 50)
    print("开始遍历阈值 0.00 ~ 1.00，步长 0.01")
    print("=" * 50)

    thresholds = np.arange(0.00, 1.01, 0.01)
    all_metrics = []

    for th in thresholds:
        th = round(th, 2)
        metrics = evaluate_with_threshold(y_true, y_pred, th)
        all_metrics.append(metrics)
        print(f"threshold = {th:.2f} | ACC={metrics['accuracy']:.3f} | AUC-ROC={metrics['auc_roc']:.3f} | AUC-PR={metrics['auc_pr']:.3f} "
              f"| MCC={metrics['mcc']:.3f} | Recall={metrics['recall']:.3f} | Spe={metrics['specificity']:.3f} | Precision={metrics['precision']:.3f}"
              f"| F1={metrics['f1']:.3f} | BAC={metrics['bac']:.3f} | AUC10%={metrics['auc10%']:.3f}")

    df = pd.DataFrame(all_metrics)
    best_idx = df['f1'].idxmax()
    best_metrics = all_metrics[best_idx]

    print("\n" + "=" * 60)
    print(f"最优阈值 (F1 最大): {best_metrics['threshold']:.2f}")
    print("=" * 60)
    for k, v in best_metrics.items():
        if k != 'threshold':
            print(f"Best Test {k}: {v:.3f}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.2f}s")


if __name__ == '__main__':
    set_seed()
    test()
