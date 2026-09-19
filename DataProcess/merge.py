import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default='../Data/train.csv', type=str,
                            required=False,
                            help="The test dataset that would be triggered")
    parser.add_argument("--vul_data_file", default=None, type=str, required=False,
                        help="The vulnerability samples located near decision boundary")
    parser.add_argument("--fix_data_file", default='../Data/train_nontovul_ATGNet.csv', type=str, required=False,
                        help="The non-vulnerability samples located near decision boundary")
    args = parser.parse_args()

    train = pd.read_csv(args.train_dataset)
    fix = pd.read_csv(args.fix_data_file)
    vul = pd.read_csv(args.vul_data_file)

    # fix 替换为 perturbed_code，target 设为 0
    mask_fix = train['index'].isin(fix['index'])
    train.loc[mask_fix, 'processed_func'] = train.loc[mask_fix, 'index'].map(fix.set_index('index')['perturbed_code'])
    train.loc[mask_fix, 'target'] = 1

    # vul 替换为 perturbed_code，target 设为 1（会覆盖 fix）
    mask_vul = train['index'].isin(vul['index'])
    train.loc[mask_vul, 'processed_func'] = train.loc[mask_vul, 'index'].map(vul.set_index('index')['perturbed_code'])
    train.loc[mask_vul, 'target'] = 0

    train.to_csv('../Data/train_poison.csv')