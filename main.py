from codes import FClassifier, ModalityRep
# from transforms.codes import event_parse
from public_function import deal_config, get_config
import pandas as pd
import argparse

def train_dataset_gaia():
    config = get_config('gaia_config.yaml')
    label_path = config['label_path']
    labels = pd.read_csv(label_path, index_col=0)
    dataset = config['dataset']
    print(dataset)

    # print('[parse]')
    # event_parse.run_parse(deal_config(config, 'parse'), labels)

    # print('[rep]')
    # ModalityRep.run_trans(deal_config(config, 'rep'), labels)

    print('[clf]')
    FClassifier.UnircaLab(deal_config(config, 'clf')).run()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gaia')
    args = parser.parse_args()
    if args.dataset == 'gaia':
        train_dataset_gaia()
    else:
        raise Exception()

if __name__ == '__main__':
    main()