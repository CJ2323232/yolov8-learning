"""把本项目 VOC2028/Annotations 中的 VOC XML 标注转为 YOLO TXT。"""
from pathlib import Path
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent
XML_DIR = BASE_DIR / "VOC2028" / "Annotations"
TXT_DIR = BASE_DIR / "VOC2028" / "labels"
CLASS_MAP = {"hat": 0, "person": 1}


def main():
    if not XML_DIR.is_dir():
        raise FileNotFoundError(f"找不到 XML 标注目录：{XML_DIR}")
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for xml_path in sorted(XML_DIR.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"图片尺寸无效：{xml_path.name}")
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in CLASS_MAP:
                print(f"跳过未知类别：{class_name}，文件：{xml_path.name}")
                continue
            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
            if not (0 <= xmin < xmax <= img_w and 0 <= ymin < ymax <= img_h):
                raise ValueError(f"检测框无效：{xml_path.name}")
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            yolo_lines.append(
                f"{CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
        (TXT_DIR / f"{xml_path.stem}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )
        count += 1
    print(f"转换完成：{count} 个 XML，YOLO 标签保存到 {TXT_DIR}")


if __name__ == "__main__":
    main()
