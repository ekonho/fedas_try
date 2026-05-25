"""
FedAS ABCD 消融实验自动运行脚本
A: 完整FedAS (PA+CS)
B: 禁用本地对齐 (--wo_local)
C: 禁用全局同步 (--wo_global)
D: 都禁用 (--wo_local --wo_global)

数据集: CIFAR-100, 其他超参数不变
"""

import os
import sys
import subprocess
import shutil
import glob
import time
from datetime import datetime

# ============================================================
#  配置
# ============================================================
PROJECT_DIR = r"E:\hqb_code\FedAS-main2"
SYSTEM_DIR  = os.path.join(PROJECT_DIR, "system")
# h5 文件实际保存在项目根目录的 results/ 下（serverbase.py 里 ../results/）
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "ABCD_results")
PYTHON = r"E:\anaconda\envs\pfllib\python.exe"

# 基础命令（参数与 main.py 中定义的一致）
BASE_CMD = f'"{PYTHON}" main.py'

# 四组实验：在基础命令上叠加消融参数
EXPERIMENTS = {
    "A": BASE_CMD,                                        # 完整FedAS
    "B": BASE_CMD + " --wo_local",                        # 禁本地对齐
    "C": BASE_CMD + " --wo_global",                       # 禁全局同步
    "D": BASE_CMD + " --wo_local --wo_global",            # 都禁
}

DESCRIPTIONS = {
    "A": "完整FedAS (PA+CS)",
    "B": "禁用本地对齐 (wo_local)",
    "C": "禁用全局同步 (wo_global)",
    "D": "禁用本地对齐+全局同步",
}


# ============================================================
#  工具函数
# ============================================================
def h5_snapshot(results_dir):
    snap = {}
    if os.path.isdir(results_dir):
        for f in glob.glob(os.path.join(results_dir, "*.h5")):
            snap[f] = os.path.getmtime(f)
    return snap


def find_new_or_modified(before, after):
    new_files = [f for f in after if f not in before]
    mod_files = [f for f in after if f in before and after[f] != before[f]]
    return sorted(new_files + mod_files)


def run_one(name, command):
    print(f"\n{'='*60}")
    print(f"  实验 {name}: {DESCRIPTIONS[name]}")
    print(f"  命令: {command}")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    before = h5_snapshot(RESULTS_DIR)

    proc = subprocess.run(command, cwd=SYSTEM_DIR, shell=True)

    if proc.returncode != 0:
        print(f"[ERROR] 实验 {name} 失败 (exit {proc.returncode})")
        return False, []

    print(f"[OK] 实验 {name} 完成")
    time.sleep(1)

    after = h5_snapshot(RESULTS_DIR)
    target_files = find_new_or_modified(before, after)

    if not target_files:
        print(f"[WARN] 未检测到新h5文件，取最新的")
        all_h5 = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.h5")),
                        key=os.path.getmtime, reverse=True)
        if all_h5:
            target_files = [all_h5[0]]

    saved = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx, src in enumerate(target_files):
        basename = os.path.basename(src)
        dst = os.path.join(OUTPUT_DIR, f"{name}_{basename}")
        if os.path.exists(dst):
            stem, ext = os.path.splitext(basename)
            dst = os.path.join(OUTPUT_DIR, f"{name}_{stem}_{idx}{ext}")
        shutil.copy2(src, dst)
        saved.append(dst)
        print(f"  -> 保存: {dst}")

    return True, saved


def compare_results():
    compare_script = os.path.join(PROJECT_DIR, "compare_results.py")
    print(f"\n{'='*60}")
    print("  调用结果对比脚本")
    print(f"{'='*60}")
    subprocess.run(
        f'"{PYTHON}" "{compare_script}"',
        cwd=PROJECT_DIR, shell=True,
    )


# ============================================================
#  主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = datetime.now()
    print(f"FedAS ABCD 消融实验开始: {start:%Y-%m-%d %H:%M:%S}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Python: {PYTHON}")
    print()
    for k, v in DESCRIPTIONS.items():
        print(f"  {k}: {v}")
    print()

    for name in ["A", "B", "C", "D"]:
        ok, saved = run_one(name, EXPERIMENTS[name])
        if not ok:
            print(f"[FATAL] 实验 {name} 失败，中止后续实验")
            sys.exit(1)

    compare_results()

    elapsed = datetime.now() - start
    print(f"\n全部完成，总耗时: {elapsed}")


if __name__ == "__main__":
    main()
