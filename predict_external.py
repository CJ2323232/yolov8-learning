"""Predict external photos with the registered model or clearly marked historical demo."""
import argparse
import sys
from run import main as run
from workflow import ROOT, select_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    model, origin = select_model(args.model)
    print(f'模型来源：{origin}；无人工真值的照片只作定性观察。')
    return run(['predict', '--model', str(model), '--source', str(ROOT/'helmet_external_test/external_test'),
                '--project', str(ROOT/'runs'), '--name', 'external_15_new', *rest])


if __name__ == '__main__':
    main()
