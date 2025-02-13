import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import argparse

def compute_eer_single_score(scores:pd.Series, labels:pd.Series) -> tuple:
    """
    Compute the Equal Error Rate (EER) for a single score.

    Args:
        scores (list or np.array): Prediction scores
        labels (list or np.array): Ground truth labels

    Returns:
        eer (float): Equal Error Rate (EER).
        threshold (float): Score threshold at which the EER occurs.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(scores) == 0 or len(labels) == 0:
        raise ValueError("Scores or labels are empty. Check your data.")
    if np.any(np.isnan(scores)):
        raise ValueError("Scores contain NaN or invalid values.")
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    threshold = thresholds[eer_index]
    
    return eer, threshold


def main(score_file:str, metadata_path:str):
    pred_df = pd.read_csv(score_file, sep=" ", header=None, names=["utt", "score"], dtype={"utt": str})

    ground_truth = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("- ")  
                utt = str(parts[1].strip())
                label = parts[4].strip().lower()  
                label = 1 if label == 'bonafide' else 0
                ground_truth.append([utt, label])

    eval_df = pd.DataFrame(ground_truth, columns=["utt", "label"])

    res_df = pd.merge(eval_df, pred_df, on='utt')

    scores = res_df['score']
    labels = res_df['label']

    eer, threshold = compute_eer_single_score(scores, labels)
    print("Final EER: {:.4f}%, Threshold: {:.4f}".format(eer * 100, threshold))

    res_df['pred'] = res_df['score'].apply(lambda x: 1 if x >= threshold else 0)
    res_df['pred'] = res_df['pred'].apply(lambda x: 'bonafide' if x == 1 else 'spoof')

    roc_auc = roc_auc_score(labels, scores)
    print("ROC AUC: {:.4f}".format(roc_auc))

    avg_precision = average_precision_score(labels, scores)
    print("Average Precision: {:.4f}".format(avg_precision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Score Evaluation")
    parser.add_argument("--score_file", type=str, help="Path to the score file", required=True)
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata file", required=True)
    args = parser.parse_args()
    main(args.score_file, args.metadata_path)
