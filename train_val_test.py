
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import (precision_recall_curve, confusion_matrix, roc_auc_score, roc_curve, auc, recall_score,
                             precision_score, f1_score, matthews_corrcoef)
import pickle
from GCAT_BCE import GCN_GAT
import numpy as np
import random
import os
import time

torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_test(labels, outputs):
    outputs = torch.sigmoid(torch.from_numpy(outputs)).numpy()
    predicted = (outputs > 0.18).astype(float)
    precision, recall, _ = precision_recall_curve(labels, outputs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()

    fpr, tpr, _ = roc_curve(labels, outputs)
    max_fpr = 0.1
    mask = fpr <= max_fpr
    fpr_filtered = np.append(fpr[mask], max_fpr)
    tpr_filtered = np.append(tpr[mask], np.interp(max_fpr, fpr, tpr))
    auc10 = auc(fpr_filtered, tpr_filtered) / max_fpr

    sensitivity = recall_score(labels, predicted)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    bac = (sensitivity + specificity) / 2

    return {
        'accuracy': (predicted == labels).mean(),
        'auc_roc': roc_auc_score(labels, outputs),
        'auc_pr': auc(recall, precision),
        'mcc': matthews_corrcoef(labels, predicted),
        'recall': recall_score(labels, predicted),
        'specificity': tn / (tn + fp),
        'precision': precision_score(labels, predicted),
        'f1': f1_score(labels, predicted),
        'bac': round(bac, 4),
        'auc10%': round(auc10, 4)
    }


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True)
            out = model(data.x, data.dist, data.adj)
            ys.append(data.y.cpu())
            preds.append(out.cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    return compute_metrics_test(y_true, y_pred)


def train():
    start_time = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # 1. 加载数据
    with open('feature/BP3_582_ordc+weighted.pkl', 'rb') as f:
        all_data = pickle.load(f)
    with open('feature/BP3_15_ordc+weighted.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # 2. 划分训练和验证集（按 8:2）
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_set = all_data[:split_idx]
    val_set = all_data[split_idx:]

    # 3. 构造 DataLoader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 4. 初始化模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Use device: {device}")
    model = GCN_GAT().to(device)
    print(f"Model device: {next(model.parameters()).device}")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    grad_clip = 1.0

    # 5. 训练
    early_stop_counter = 0
    patience = 10
    best_val = 0
    best_val_loss = float('inf')
    best_model_path = "best_model_GCAT-BCE.pt"

    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        for idx, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    out = model(data.x, data.dist, data.adj)
                    loss = F.binary_cross_entropy_with_logits(out, data.y.float())
            else:
                out = model(data.x, data.dist, data.adj)
                loss = F.binary_cross_entropy_with_logits(out, data.y.float())


            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at epoch {epoch}, batch {idx}, skip backward")
                optimizer.zero_grad()
                continue

            # 反向传播
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()

        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(data.x, data.dist, data.adj)
                        loss = F.binary_cross_entropy_with_logits(out, data.y.float())
                else:
                    out = model(data.x, data.dist, data.adj)
                    loss = F.binary_cross_entropy_with_logits(out, data.y.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch:03d} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"Val {k}: {v:.3f}", end=' | ')
        print()

        # Early stopping 判断逻辑
        if val_loss < best_val_loss or val_metrics['auc_pr'] > best_val:
            best_val = val_metrics['auc_pr']
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=False)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                print(f"Best Val AUC-PR: {best_val:.3f}, Best Val Loss: {best_val_loss:.4f}")
                break

    # 6. 加载最佳模型，测试
    print("\n[Testing best model on test set]")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.3f}")

    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.2f}s")


if __name__ == '__main__':
    set_seed()
    train()
