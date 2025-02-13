import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import argparse

def compute_eer(bonafide_scores, spoof_scores):
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        raise ValueError("bonafide_scores or spoof_scores is empty. Check your data filtering.")
    
    if np.any(np.isnan(bonafide_scores)) or np.any(np.isnan(spoof_scores)):
        raise ValueError("bonafide_scores or spoof_scores contains NaN or invalid values.")
    
    y_true = [1] * len(bonafide_scores) + [0] * len(spoof_scores)
    y_scores = list(bonafide_scores) + list(spoof_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_index]
    eer = fpr[eer_index]
    
    print("EER: {:.4f}, Threshold: {:.4f}".format(eer, eer_threshold))
    
    return eer, eer_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EER Calculation")
    parser.add_argument("--score_file", type=str, help="Path to the prediction file")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata file")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.score_file, sep=" ", header=None)
    pred_df.columns = ["utt", "score1", "score2"]
    pred_df['utt'] = pred_df['utt'].str.replace('.wav', '')
    pred_df['score'] = pred_df['score2'] - pred_df['score1']
    pred_df['pred'] = pred_df['score'].apply(lambda x: 'spoof' if x >= 0 else 'bonafide')

    ground_truth = []
    with open(args.metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("- ")
                utt = parts[1].strip()
                label = parts[4].strip()
                ground_truth.append([utt, label])
    eval_df = pd.DataFrame(ground_truth, columns=["utt", "label"])

    res_df = pd.merge(eval_df, pred_df, on='utt')
    spoof_scores = res_df[res_df['label'] == 'spoof']['score']
    bonafide_scores = res_df[res_df['label'] == 'bonafide']['score']

    # Calculate EER
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print("EER: {:.4f}%, threshold: {:.4f}".format(eer*100, threshold))

    res_df['pred'] = res_df['score'].apply(lambda x: 'spoof' if x < threshold else 'bonafide')

    # Convert labels to numeric (1 for bonafide, 0 for spoof) for AUC and AP calculations
    res_df['label_num'] = res_df['label'].apply(lambda x: 1 if x == 'bonafide' else 0)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(res_df['label_num'], res_df['score'])
    print("ROC AUC: {:.4f}".format(roc_auc))

    # Calculate Average Precision
    avg_precision = average_precision_score(res_df['label_num'], res_df['score'])
    print("Average Precision: {:.4f}".format(avg_precision))