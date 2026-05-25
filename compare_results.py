"""
对比 ABCD 消融实验的 h5 结果文件。
A: 完整FedAS (PA+CS)    B: 禁用本地对齐    C: 禁用全局同步    D: 都禁用

h5 keys: rs_test_acc, rs_test_auc, rs_train_loss

用法:
  python compare_results.py [结果目录]
  不传目录则默认 ABCD_results/
"""

import os
import sys
import glob
import h5py
import numpy as np
from datetime import datetime

DEFAULT_DIR = os.path.join(r"E:\hqb_code\FedAS-main2", "ABCD_results")

EXP_LABELS = {
    "A": "完整FedAS(PA+CS)",
    "B": "禁用本地对齐(wo_local)",
    "C": "禁用全局同步(wo_global)",
    "D": "都禁用",
}


def to_scalar(v):
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return float('nan')
        return float(v[-1]) if v.ndim >= 1 else float(v)
    return float(v)


def to_array(v):
    if isinstance(v, np.ndarray):
        return v.flatten()
    return np.array([v])


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR

    print(f"\n{'='*60}")
    print("  FedAS ABCD 消融实验结果对比")
    print(f"  目录: {output_dir}")
    print(f"{'='*60}")

    h5_files = sorted(glob.glob(os.path.join(output_dir, "*.h5")))
    if not h5_files:
        print("  没有找到任何 h5 结果文件！")
        return

    results = {}
    for fpath in h5_files:
        fname = os.path.basename(fpath)
        exp_name = fname.split("_")[0]
        if exp_name not in EXP_LABELS:
            continue
        try:
            with h5py.File(fpath, "r") as hf:
                results[exp_name] = {
                    "file":       fname,
                    "test_acc":   np.array(hf["rs_test_acc"]),
                    "test_auc":   np.array(hf["rs_test_auc"]),
                    "train_loss": np.array(hf["rs_train_loss"]),
                }
        except Exception as e:
            print(f"  读取 {fname} 失败: {e}")

    if not results:
        return

    # ------ 汇总表格 ------
    print(f"\n{'实验':<4} {'说明':<22} {'最终TestAcc':>12} {'最高TestAcc':>12} "
          f"{'最终TestAUC':>12} {'最终TrainLoss':>14}")
    print("-" * 82)

    best_acc = {}
    for name in sorted(results):
        r = results[name]
        label = EXP_LABELS.get(name, name)
        fa  = to_scalar(r["test_acc"])
        pa  = float(np.max(r["test_acc"]))
        auc = to_scalar(r["test_auc"])
        tl  = to_scalar(r["train_loss"])
        best_acc[name] = pa
        print(f"{name:<4} {label:<22} {fa:>12.4f} {pa:>12.4f} "
              f"{auc:>12.4f} {tl:>14.4f}")

    # ------ 消融对比 ------
    print(f"\n--- 消融对比 (相对于A的差异) ---")
    if "A" in results:
        a_final = to_scalar(results["A"]["test_acc"])
        a_peak  = best_acc["A"]
        for name in ["B", "C", "D"]:
            if name in results:
                nf = to_scalar(results[name]["test_acc"])
                np_ = best_acc[name]
                label = EXP_LABELS.get(name, name)
                print(f"  A vs {name} ({label}):")
                print(f"    最终TestAcc: {a_final:.4f} vs {nf:.4f}  "
                      f"(Δ = {nf-a_final:+.4f})")
                print(f"    最高TestAcc: {a_peak:.4f} vs {np_:.4f}  "
                      f"(Δ = {np_-a_peak:+.4f})")
    else:
        print("  (缺少A组结果，无法对比)")

    # ------ 保存汇总文件 ------
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"FedAS ABCD 消融实验汇总  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'实验':<4} {'说明':<22} {'最终TestAcc':>12} {'最高TestAcc':>12}\n")
        f.write("-" * 50 + "\n")
        for name in sorted(results):
            r = results[name]
            label = EXP_LABELS.get(name, name)
            f.write(f"{name:<4} {label:<22} "
                    f"{to_scalar(r['test_acc']):>12.4f} "
                    f"{float(np.max(r['test_acc'])):>12.4f}\n")
        f.write("\n每轮 TestAcc 曲线:\n")
        for name in sorted(results):
            curve = to_array(results[name]["test_acc"])
            label = EXP_LABELS.get(name, name)
            f.write(f"\n{name} ({label}): "
                    + ", ".join(f"{v:.4f}" for v in curve) + "\n")

    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
