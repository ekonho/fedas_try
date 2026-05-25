"""
FedAS ABCD 消融实验自动运行脚本（多轮版）
A: 完整FedAS (PA+CS)
B: 禁用本地对齐 (--wo_local)
C: 禁用全局同步 (--wo_global)
D: 都禁用 (--wo_local --wo_global)

目录结构:
  ABCD_results/pretrained/round1/   round2/   ...
  ABCD_results/notpretrained/round1/ round2/  ...
"""

import os
import sys
import subprocess
import shutil
import glob
import time
from datetime import datetime

# ============================================================
#  配置（需要改这里）
# ============================================================
PROJECT_DIR    = r"E:\hqb_code\FedAS-main2"
SYSTEM_DIR     = os.path.join(PROJECT_DIR, "system")
RESULTS_DIR    = os.path.join(PROJECT_DIR, "results")       # h5 保存位置
OUTPUT_ROOT    = os.path.join(PROJECT_DIR, "ABCD_results")  # 输出根目录
PYTHON         = r"E:\anaconda\envs\pfllib\python.exe"

NUM_ROUNDS     = 1            # 跑几轮 ABCD
USE_PRETRAINED = True        # True = 预训练ResNet, False = 非预训练,main.py的model_str == "resnet"判断

# 根据是否预训练选择输出目录
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "pretrained" if USE_PRETRAINED else "notpretrained")

# 基础命令
BASE_CMD = f'"{PYTHON}" main.py'

# 四组实验
EXPERIMENTS = {
    "A": BASE_CMD,
    "B": BASE_CMD + " --wo_local",
    "C": BASE_CMD + " --wo_global",
    "D": BASE_CMD + " --wo_local --wo_global",
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


def clear_results_dir():
    """清空 results 目录，防止下一轮的 h5 覆盖"""
    if os.path.isdir(RESULTS_DIR):
        for f in glob.glob(os.path.join(RESULTS_DIR, "*.h5")):
            os.remove(f)


def run_one(name, command, round_dir):
    """运行单个实验，h5 复制到 round_dir"""
    print(f"\n{'='*60}")
    print(f"  实验 {name}: {DESCRIPTIONS[name]}")
    print(f"  命令: {command}")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    before = h5_snapshot(RESULTS_DIR)

    proc = subprocess.run(command, cwd=SYSTEM_DIR, shell=True)
    if proc.returncode != 0:
        print(f"[ERROR] 实验 {name} 失败 (exit {proc.returncode})")
        return False

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

    os.makedirs(round_dir, exist_ok=True)
    for idx, src in enumerate(target_files):
        basename = os.path.basename(src)
        dst = os.path.join(round_dir, f"{name}_{basename}")
        if os.path.exists(dst):
            stem, ext = os.path.splitext(basename)
            dst = os.path.join(round_dir, f"{name}_{stem}_{idx}{ext}")
        shutil.copy2(src, dst)
        print(f"  -> 保存: {dst}")

    return True


def compare_round(round_dir):
    """对比单轮结果"""
    compare_script = os.path.join(PROJECT_DIR, "compare_results.py")
    print(f"\n{'='*60}")
    print(f"  对比结果: {round_dir}")
    print(f"{'='*60}")
    subprocess.run(
        f'"{PYTHON}" "{compare_script}" "{round_dir}"',
        cwd=PROJECT_DIR, shell=True,
    )


# ============================================================
#  主流程
# ============================================================
def get_next_round_num():
    """扫描已有 round 文件夹，返回下一个编号"""
    existing = []
    if os.path.isdir(OUTPUT_DIR):
        for d in os.listdir(OUTPUT_DIR):
            if d.startswith("round"):
                try:
                    existing.append(int(d[5:]))
                except ValueError:
                    pass
    return max(existing, default=0) + 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = datetime.now()
    mode = "预训练ResNet" if USE_PRETRAINED else "非预训练ResNet"
    next_round = get_next_round_num()

    print(f"FedAS ABCD 消融实验开始: {start:%Y-%m-%d %H:%M:%S}")
    print(f"模式: {mode}")
    print(f"本轮从 round{next_round} 开始，共跑 {NUM_ROUNDS} 轮")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Python: {PYTHON}")
    print()

    for i in range(NUM_ROUNDS):
        rnd = next_round + i
        round_dir = os.path.join(OUTPUT_DIR, f"round{rnd}")
        print(f"\n{'#'*60}")
        print(f"  round{rnd}  ({i+1}/{NUM_ROUNDS})  ->  {round_dir}")
        print(f"{'#'*60}")

        os.makedirs(round_dir, exist_ok=True)
        clear_results_dir()   # 清空 results，确保本轮 h5 不被污染

        for name in ["A", "B", "C", "D"]:
            ok = run_one(name, EXPERIMENTS[name], round_dir)
            if not ok:
                print(f"[FATAL] round{rnd} 实验 {name} 失败，中止")
                sys.exit(1)

        compare_round(round_dir)

    elapsed = datetime.now() - start
    print(f"\n全部完成，共 {NUM_ROUNDS} 轮，总耗时: {elapsed}")


if __name__ == "__main__":
    main()
