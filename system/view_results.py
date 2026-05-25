import h5py
import numpy as np
import csv
import sys
import os
import glob

# ● 三种功能都好了。
#
#   用法：
#
#   # 1. 查看单个文件（完整每轮数据 + 元信息）
#   python view_results.py Cifar100_FedAS_test_0.h5
#
#   # 2. 对比多个文件（并排显示 acc 和 loss）
#   python view_results.py file1.h5 file2.h5 file3.h5
#
#   # 3. 导出 CSV
#   python view_results.py --csv Cifar100_FedAS_test_0.h5
#   # 输出到 results/Cifar100_FedAS_test_0.csv
#
#   # 4. 查看 results/ 下所有文件对比
#   python view_results.py --all
#
#   # 无参数：列出所有可用文件
#   python view_results.py
#
#   多文件对比时会输出：每轮 acc/loss 并排表 + 最佳轮次汇总 + 元信息摘要。CSV 会自动从文件名解析 dataset/algorithm
#   等信息写入首行。

def parse_filename(filepath):
    """Parse metadata from filename: {dataset}_{algorithm}_{goal}_{times}.h5"""
    name = os.path.splitext(os.path.basename(filepath))[0]
    parts = name.split('_')
    info = {}
    if len(parts) >= 4:
        info['dataset'] = parts[0]
        info['algorithm'] = parts[1]
        info['goal'] = parts[2]
        info['times'] = parts[3]
    elif len(parts) >= 3:
        info['dataset'] = parts[0]
        info['algorithm'] = parts[1]
        info['goal'] = parts[2]
    return info


def read_h5(filepath):
    """Read h5 file and return data dict + metadata."""
    f = h5py.File(filepath, 'r')
    data = {}
    for key in f.keys():
        data[key] = f[key][()]
    f.close()
    meta = parse_filename(filepath)
    meta['file'] = os.path.basename(filepath)
    return data, meta


def print_single(filepath):
    """Print full details of a single h5 file."""
    data, meta = read_h5(filepath)

    print(f"\nFile: {filepath}")
    print(f"{'='*60}")
    print(f"  Dataset:   {meta.get('dataset', 'N/A')}")
    print(f"  Algorithm: {meta.get('algorithm', 'N/A')}")
    print(f"  Goal:      {meta.get('goal', 'N/A')}")
    print(f"  Times:     {meta.get('times', 'N/A')}")

    for key in sorted(data.keys()):
        arr = data[key]
        print(f"\n  --- {key} (shape: {arr.shape}) ---")

        if arr.size == 0:
            print("    (empty)")
            continue

        print(f"    max: {np.max(arr):.6f}  min: {np.min(arr):.6f}")

        if 'acc' in key.lower() and arr.size > 0:
            best = np.argmax(arr)
            print(f"    best: round {best} -> {arr[best]:.4f}")

        print(f"\n    {'round':>5s}  {'value':>10s}")
        print(f"    {'-----':>5s}  {'-----':>10s}")
        for i, v in enumerate(arr):
            print(f"    {i:5d}  {v:10.4f}")

    return data, meta


def compare_files(filepaths):
    """Compare multiple h5 files side by side (acc and loss)."""
    results = []
    for fp in filepaths:
        data, meta = read_h5(fp)
        results.append((meta, data))

    # --- Accuracy comparison ---
    acc_key = 'rs_test_acc'
    has_acc = all(acc_key in d and d[acc_key].size > 0 for _, d in results)
    if has_acc:
        max_rounds = max(d[acc_key].size for _, d in results)
        print(f"\n{'='*80}")
        print(f"Test Accuracy Comparison")
        print(f"{'='*80}")

        # header
        header = f"{'round':>5s}"
        for meta, _ in results:
            label = meta.get('file', '?')[:25]
            header += f"  {label:>25s}"
        print(header)
        print('-' * (5 + 27 * len(results)))

        for r in range(max_rounds):
            line = f"{r:5d}"
            for meta, data in results:
                arr = data[acc_key]
                if r < arr.size:
                    line += f"  {arr[r]:25.4f}"
                else:
                    line += f"  {'---':>25s}"
            print(line)

        # best summary
        print('-' * (5 + 27 * len(results)))
        line = f"{'best':>5s}"
        for meta, data in results:
            arr = data[acc_key]
            best = np.argmax(arr)
            line += f"  {arr[best]:21.4f}@R{best:<2d}"
        print(line)

    # --- Loss comparison ---
    loss_key = 'rs_train_loss'
    has_loss = all(loss_key in d and d[loss_key].size > 0 for _, d in results)
    if has_loss:
        max_rounds = max(d[loss_key].size for _, d in results)
        print(f"\n{'='*80}")
        print(f"Train Loss Comparison")
        print(f"{'='*80}")

        header = f"{'round':>5s}"
        for meta, _ in results:
            label = meta.get('file', '?')[:25]
            header += f"  {label:>25s}"
        print(header)
        print('-' * (5 + 27 * len(results)))

        for r in range(max_rounds):
            line = f"{r:5d}"
            for meta, data in results:
                arr = data[loss_key]
                if r < arr.size:
                    line += f"  {arr[r]:25.4f}"
                else:
                    line += f"  {'---':>25s}"
            print(line)

        print('-' * (5 + 27 * len(results)))
        line = f"{'best':>5s}"
        for meta, data in results:
            arr = data[loss_key]
            best = np.argmin(arr)
            line += f"  {arr[best]:21.4f}@R{best:<2d}"
        print(line)

    # --- Metadata summary ---
    print(f"\n{'='*80}")
    print(f"Metadata Summary")
    print(f"{'='*80}")
    for meta, data in results:
        acc = data.get('rs_test_acc', np.array([]))
        loss = data.get('rs_train_loss', np.array([]))
        acc_str = f"{np.max(acc):.4f}@R{np.argmax(acc)}" if acc.size > 0 else "N/A"
        loss_str = f"{np.min(loss):.4f}@R{np.argmin(loss)}" if loss.size > 0 else "N/A"
        print(f"  {meta.get('file','?'):<35s}  dataset={meta.get('dataset','?'):<10s}  "
              f"algo={meta.get('algorithm','?'):<10s}  best_acc={acc_str:<15s}  best_loss={loss_str}")


def export_csv(filepath, outpath=None):
    """Export h5 data to CSV."""
    data, meta = read_h5(filepath)
    if outpath is None:
        outpath = os.path.splitext(filepath)[0] + '.csv'

    keys = sorted(data.keys())
    # find max rounds
    max_rounds = max((data[k].size for k in keys), default=0)

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        # metadata row
        writer.writerow([f"# dataset={meta.get('dataset','')}, "
                         f"algorithm={meta.get('algorithm','')}, "
                         f"goal={meta.get('goal','')}, "
                         f"times={meta.get('times','')}"])
        # header
        writer.writerow(['round'] + keys)
        # data
        for r in range(max_rounds):
            row = [r]
            for k in keys:
                arr = data[k]
                row.append(f"{arr[r]:.6f}" if r < arr.size else '')
            writer.writerow(row)

    print(f"Exported to: {outpath}")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    if len(sys.argv) < 2:
        # no args: list all files
        files = sorted(glob.glob(os.path.join(results_dir, '*.h5')), key=os.path.getmtime)
        if not files:
            print("No h5 files found in results/")
            sys.exit(1)
        print("Available result files:")
        for i, f in enumerate(files):
            print(f"  [{i}] {os.path.basename(f)}")
        print(f"\nUsage:")
        print(f"  python view_results.py                   # view latest")
        print(f"  python view_results.py <file>             # view one file")
        print(f"  python view_results.py <file1> <file2>    # compare files")
        print(f"  python view_results.py --csv <file>       # export to CSV")
        print(f"  python view_results.py --all              # view all files")
        sys.exit(0)

    if sys.argv[1] == '--csv':
        fp = sys.argv[2] if len(sys.argv) > 2 else sorted(
            glob.glob(os.path.join(results_dir, '*.h5')), key=os.path.getmtime)[-1]
        if not os.path.isabs(fp):
            fp = os.path.join(results_dir, fp)
        export_csv(fp)

    elif sys.argv[1] == '--all':
        files = sorted(glob.glob(os.path.join(results_dir, '*.h5')), key=os.path.getmtime)
        if len(files) > 1:
            compare_files(files)
        elif files:
            print_single(files[0])

    elif len(sys.argv) == 2:
        fp = sys.argv[1]
        if not os.path.isabs(fp):
            fp = os.path.join(results_dir, fp)
        print_single(fp)

    else:
        fps = []
        for arg in sys.argv[1:]:
            fp = arg
            if not os.path.isabs(fp):
                fp = os.path.join(results_dir, fp)
            fps.append(fp)
        compare_files(fps)
