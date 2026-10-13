"""Train a fresh helmet baseline on the audited dataset; importing is safe."""
import sys
from run import main as run
from workflow import ROOT


def main(argv=None):
    pretrained = str(ROOT/'yolov8n.pt') if (ROOT/'yolov8n.pt').is_file() else 'yolov8n.pt'
    return run(['train', '--model', pretrained, '--data', str(ROOT/'helmet_dataset_clean/data.yaml'),
                '--epochs', '50', '--imgsz', '640', '--batch', '4',
                '--optimizer', 'AdamW', '--lr0', '0.001667', '--seed', '0',
                '--project', str(ROOT/'runs'), '--name', 'helmet_clean',
                *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
