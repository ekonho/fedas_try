"""
PA (Parameter Alignment) 效果分析工具
用法：
  python pa_analysis.py --mode drift    # 测量原型漂移（cosine similarity）
  python pa_analysis.py --mode consistency  # 追踪特征一致性
  python pa_analysis.py --mode ablation # 消融实验对比（需要两组实验结果）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from flcore.trainmodel.resnet import resnet18
from flcore.trainmodel.models import BaseHeadSplit
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader


def load_model(num_classes=100):
    model = resnet18(num_classes=num_classes, has_bn=True, bn_block_num=4)
    head = model.fc
    model.fc = nn.Identity()
    return BaseHeadSplit(model, head)


def get_prototypes(model, dataloader, device, num_classes):
    """Extract class prototypes from backbone."""
    model.eval()
    proto_sums = [None] * num_classes
    proto_counts = [0] * num_classes

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = model.base(x)
            for label in y.unique().cpu().numpy():
                mask = (y == label)
                class_features = features[mask].mean(dim=0)
                if proto_sums[label] is None:
                    proto_sums[label] = class_features.clone()
                else:
                    proto_sums[label] += class_features
                proto_counts[label] += 1

    prototypes = []
    for i in range(num_classes):
        if proto_counts[i] > 0:
            prototypes.append((i, proto_sums[i] / proto_counts[i]))
    return prototypes


def cosine_similarity(p1, p2):
    """Cosine similarity between two prototype lists (matched by class)."""
    p1_dict = {cls: feat for cls, feat in p1}
    p2_dict = {cls: feat for cls, feat in p2}
    common = set(p1_dict.keys()) & set(p2_dict.keys())
    if not common:
        return 0.0
    sims = []
    for cls in common:
        f1 = p1_dict[cls].flatten()
        f2 = p2_dict[cls].flatten()
        sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        sims.append(sim)
    return np.mean(sims)


def measure_alignment_effect(dataset='Cifar100', num_classes=100, num_clients=20, device='cpu'):
    """
    模拟一轮 PA 过程，测量对齐前后 prototype 的变化。
    """
    print("=" * 60)
    print("PA 效果分析：原型漂移测量")
    print("=" * 60)

    # 模拟两代模型：local（旧）和 global（新）
    local_model = load_model(num_classes).to(device)
    global_model = load_model(num_classes).to(device)

    # 取几个 client 的数据
    selected_clients = [0, 5, 10, 15]
    batch_size = 16

    print(f"\n{'client':>8s}  {'before_align':>14s}  {'after_align':>14s}  {'improvement':>12s}")
    print(f"{'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}")

    all_before = []
    all_after = []

    for cid in selected_clients:
        try:
            train_data = read_client_data(dataset, cid, is_train=True)
            loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=False)
        except Exception as e:
            print(f"  Client {cid}: data not found, skipping")
            continue

        # 本地旧模型的 prototypes
        local_protos = get_prototypes(local_model, loader, device, num_classes)

        # 对齐前：global model 的 prototypes
        global_before = get_prototypes(global_model, loader, device, num_classes)
        sim_before = cosine_similarity(local_protos, global_before)

        # 模拟 PA 对齐
        alignment_optimizer = torch.optim.SGD(global_model.base.parameters(), lr=0.001)
        alignment_loss_fn = nn.MSELoss()

        local_proto_dict = {cls: feat for cls, feat in local_protos}

        global_model.train()
        for _ in range(1):
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                global_features = global_model.base(x_batch)
                loss = 0
                for label in y_batch.unique().cpu().numpy():
                    if label in local_proto_dict:
                        mask = (y_batch == label)
                        loss += alignment_loss_fn(global_features[mask], local_proto_dict[label])
                if isinstance(loss, int) or torch.isnan(loss):
                    continue
                alignment_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(global_model.base.parameters(), max_norm=10.0)
                alignment_optimizer.step()

        # 对齐后：global model 的 prototypes
        global_model.eval()
        global_after = get_prototypes(global_model, loader, device, num_classes)
        sim_after = cosine_similarity(local_protos, global_after)

        improvement = sim_after - sim_before
        all_before.append(sim_before)
        all_after.append(sim_after)

        print(f"  {cid:>5d}  {sim_before:14.4f}  {sim_after:14.4f}  {improvement:+12.4f}")

    print(f"{'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}")
    avg_before = np.mean(all_before)
    avg_after = np.mean(all_after)
    print(f"  {'avg':>5s}  {avg_before:14.4f}  {avg_after:14.4f}  {avg_after-avg_before:+12.4f}")

    print(f"\n结论：")
    print(f"  对齐前 local/global prototype cosine sim: {avg_before:.4f}")
    print(f"  对齐后 local/global prototype cosine sim: {avg_after:.4f}")
    print(f"  改善: {(avg_after-avg_before)*100:.2f}%")

    if avg_after > avg_before + 0.01:
        print(f"  → PA 有效拉近了 global backbone 与 local prototype 的距离")
    else:
        print(f"  → PA 效果不明显（随机初始化下本就发散，改善有限）")


def analyze_feature_consistency(dataset='Cifar100', num_classes=100, device='cpu'):
    """
    分析 PA 对特征一致性的影响：
    - 相同类的样本，对齐前后的特征是否更聚集？
    """
    print("\n" + "=" * 60)
    print("PA 效果分析：特征聚类一致性")
    print("=" * 60)

    model = load_model(num_classes).to(device)

    try:
        train_data = read_client_data(dataset, 0, is_train=True)
        loader = DataLoader(train_data, batch_size=32, drop_last=True, shuffle=False)
    except Exception as e:
        print(f"  Data not found: {e}")
        return

    # 提取特征
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = model.base(x)
            all_features.append(features.cpu())
            all_labels.append(y)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # 计算类内距离 vs 类间距离
    unique_labels = labels.unique()[:10]  # 取前 10 类
    intra_dists = []
    inter_dists = []

    for i, l1 in enumerate(unique_labels):
        f1 = features[labels == l1]
        if len(f1) < 2:
            continue
        # 类内距离
        centroid1 = f1.mean(dim=0)
        intra = torch.norm(f1 - centroid1, dim=1).mean().item()
        intra_dists.append(intra)

        # 类间距离
        for l2 in unique_labels[i+1:]:
            f2 = features[labels == l2]
            centroid2 = f2.mean(dim=0)
            inter = torch.norm(centroid1 - centroid2).item()
            inter_dists.append(inter)

    avg_intra = np.mean(intra_dists) if intra_dists else 0
    avg_inter = np.mean(inter_dists) if inter_dists else 0

    print(f"\n  平均类内距离 (intra-class):  {avg_intra:.4f}")
    print(f"  平均类间距离 (inter-class):  {avg_inter:.4f}")
    print(f"  类间/类内比值:               {avg_inter/avg_intra:.2f}")

    print(f"\n  → 比值越大，特征越容易分类")
    print(f"  → PA 的目的：让 global backbone 生成的特征与 local 一致，")
    print(f"    即减小对齐后的类内距离（更聚集）")


def ablation_compare():
    """对比有无 PA 的实验结果。"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    print("\n" + "=" * 60)
    print("PA 消融实验对比")
    print("=" * 60)
    print("\n请先分别运行以下两组实验：")
    print(f"  1. 带 PA:  python main.py ... -algo FedAS")
    print(f"  2. 无 PA:  python main.py ... -algo FedAS --wo_local")
    print(f"\n然后将两个 h5 文件放到 results/ 目录，重新运行：")
    print(f"  python view_results.py file_with_PA.h5 file_without_PA.h5")
    print(f"\n预期分析维度：")
    print(f"  - 最佳 acc 差异：PA 预期提升 1-3%")
    print(f"  - 收敛速度：PA 预期前 20 轮优势更明显")
    print(f"  - 过拟合程度：PA 可能减少过拟合（特征更一致）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['drift', 'consistency', 'ablation', 'all'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='Cifar100')
    args = parser.parse_args()

    if args.mode in ('drift', 'all'):
        measure_alignment_effect(args.dataset, 100, 20, args.device)

    if args.mode in ('consistency', 'all'):
        analyze_feature_consistency(args.dataset, 100, args.device)

    if args.mode in ('ablation', 'all'):
        ablation_compare()
