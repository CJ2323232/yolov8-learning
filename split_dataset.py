"""将 SHWD 图片和已转换的 YOLO 标签划分为训练/验证/测试集。

运行：python3 split_dataset.py
脚本复制文件，不修改原始 VOC2028 数据。
"""
from collections import Counter
from pathlib import Path
import random
import shutil

# 数据集放在仓库目录下；若位置不同，修改这里。
BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "VOC2028"
IMAGE_DIR = SOURCE / "JPEGImages"
LABEL_DIR = SOURCE / "labels"
OUTPUT = BASE_DIR / "helmet_dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASS_IDS = {0, 1}  # 假设之前的映射为 hat:0, person:1


def main():
    for folder in (IMAGE_DIR, LABEL_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到文件夹：{folder}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"输出目录已存在：{OUTPUT}\n"
            "为防止覆盖之前划分的数据，请先检查该目录，然后改 OUTPUT 路径或自行备份后删除。"
        )

    images = sorted(p for p in IMAGE_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"图片文件夹中没有找到图片：{IMAGE_DIR}")

    # 必须保证一张图片对应一个同名标签，避免静默丢失训练数据。
    missing = [p.name for p in images if not (LABEL_DIR / (p.stem + ".txt")).is_file()]
    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 张图片找不到对应 TXT，前 10 个：{missing[:10]}\n"
            "请先检查 XML→TXT 是否全部转换成功。"
        )

    if len(set(p.stem for p in images)) != len(images):
        raise RuntimeError("发现图片重名但扩展名不同（例如 001.jpg 和 001.png），请先处理。")

    # 检查每个标注的格式及类别，空 TXT 允许作为无目标图片。
    image_counts = {}
    for img in images:
        label = LABEL_DIR / (img.stem + ".txt")
        classes = Counter()
        for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"标签格式不正确：{label.name} 第 {line_no} 行：{line}")
            try:
                values = [float(x) for x in parts]
            except ValueError as exc:
                raise ValueError(f"标签存在非数字：{label.name} 第 {line_no} 行") from exc
            class_id = int(values[0])
            if values[0] != class_id or class_id not in VALID_CLASS_IDS:
                raise ValueError(f"未知类别编号：{label.name} 第 {line_no} 行：{parts[0]}")
            x, y, w, h = values[1:]
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"框坐标超出范围：{label.name} 第 {line_no} 行：{line}")
            classes[class_id] += 1
        image_counts[img] = classes

    # 固定随机种子，让重新运行时划分可复现。
    rng = random.Random(SEED)
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    groups = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for subset, group in groups.items():
        (OUTPUT / "images" / subset).mkdir(parents=True)
        (OUTPUT / "labels" / subset).mkdir(parents=True)
        counts = Counter()
        for img in group:
            label = LABEL_DIR / (img.stem + ".txt")
            shutil.copy2(img, OUTPUT / "images" / subset / img.name)
            shutil.copy2(label, OUTPUT / "labels" / subset / label.name)
            counts.update(image_counts[img])
        print(f"{subset:5}：{len(group):5} 张图片；hat={counts[0]} 个框，person={counts[1]} 个框")

    # 使用绝对路径，不受从哪个目录执行脚本影响。
    (OUTPUT / "data.yaml").write_text(
        f'path: "{OUTPUT}"\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'names:\n'
        '  0: hat\n'
        '  1: person\n',
        encoding="utf-8",
    )
    print(f"\n完成，共 {n} 张。数据集：{OUTPUT}")
    print(f"YOLO 配置：{OUTPUT / 'data.yaml'}")
    print("提示：本脚本只进行随机划分；如有同一视频的连续帧，应先按拍摄场景分组，以避免数据泄漏。")


if __name__ == "__main__":
    main()
