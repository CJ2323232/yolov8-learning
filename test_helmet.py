"""Evaluate the explicitly registered training artifact, never a hard-coded old run."""
import argparse
import sys
from run import main as run
from workflow import ROOT, latest_training, select_model, verify_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--data')
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    model, origin = select_model(args.model, require_training=True)
    if origin == 'registered_training':
        record = latest_training()
        data = args.data or record['data']
        identity = verify_dataset(data)
        if identity.get('manifest_sha256') != record['dataset']['manifest_sha256']:
            raise ValueError('评价数据与已登记训练划分不同，停止自动评价')
    else:
        data = args.data or str(ROOT/'helmet_dataset_clean/data.yaml')
    return run(['val', '--model', str(model), '--data', str(data), '--split', 'test',
                '--imgsz', '640', '--batch', '4', '--project', str(ROOT/'runs'),
                '--name', 'helmet_clean_test', *rest])


if __name__ == '__main__':
    main()
