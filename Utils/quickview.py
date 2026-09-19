import pandas as pd
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--triggered_result", default=None, type=str, required=True,
                        help="The detection result with triggered inputs")
    parser.add_argument("--pure_result", default=None, type=str, required=True,
                        help="The detection result with pure inputs")
    args = parser.parse_args()

    pure_result = pd.read_csv(args.pure_result)
    triggered_result = pd.read_csv(args.triggered_result)

    print(f'triggered data size: {len(triggered_result)}, vul: {len(triggered_result[triggered_result["y_true"]==1])}, non-vul: {len(triggered_result[triggered_result["y_true"]==0])}')

    def TP_filter(row):
        return row['y_true'] == 1 and row['y_pred'] == True
        
    def TN_filter(row):
        return row['y_true'] == 0 and row['y_pred'] == False
        
    def vul_AS_filter(row):
        return row['y_true'] == 1 and row['y_pred'] == False

    def nv_AS_filter(row):
        return row['y_true'] == 0 and row['y_pred'] == True

    TP_pure_result = pure_result[pure_result.apply(TP_filter, axis=1)].iloc[:,:2].rename(columns={'logits': 'original_logits'})
    TN_pure_result = pure_result[pure_result.apply(TN_filter, axis=1)].iloc[:,:2].rename(columns={'logits': 'original_logits'})
    
    triggered_vul = triggered_result[triggered_result['index'].isin(TP_pure_result['index'])]
    triggered_nv = triggered_result[triggered_result['index'].isin(TN_pure_result['index'])]
    
    triggered_vul_attack_succeed = triggered_vul[triggered_vul.apply(vul_AS_filter, axis=1)]
    triggered_nv_attack_succeed = triggered_nv[triggered_nv.apply(nv_AS_filter, axis=1)]
    
    print(f'Attack Succeed triggered vulnerable samples {len(triggered_vul_attack_succeed)}/{len(triggered_vul)}, ASR: {len(triggered_vul_attack_succeed)/len(triggered_vul):.4f}')
    print(f'Attack Succeed triggered non-vulnerable samples {len(triggered_nv_attack_succeed)}/{len(triggered_nv)}, ASR: {len(triggered_nv_attack_succeed)/len(triggered_nv):.4f}')


if __name__ == "__main__":
    main()