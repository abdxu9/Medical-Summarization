import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_comparison_charts(results_dir: str, output_dir: str):
    """
    Scans a directory for aggregated evaluation CSV files, combines them,
    and generates comparative bar charts for all metrics.
    """
    search_pattern = os.path.join(results_dir, "*_evaluation_aggregated_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"Error: No '*_evaluation_aggregated_*.csv' files found in '{results_dir}'.")
        return

    print(f"Found {len(csv_files)} result files to plot.")
    
    # Read and combine all result files
    df_list = [pd.read_csv(f) for f in csv_files]
    results_df = pd.concat(df_list, ignore_index=True)
    
    # --- 1. Plot BERTScore Comparison ---
    bertscore_df = results_df.sort_values(by="bert_score_f1_mean", ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))

    models = bertscore_df['model'].tolist()
    metrics = ['bert_score_precision_mean', 'bert_score_recall_mean', 'bert_score_f1_mean']
    metric_labels = ['Precision', 'Recall', 'F1']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = bertscore_df[metric].tolist()
        errors = bertscore_df[metric.replace('mean', 'std')].tolist()
        plt.bar(x + i * width, values, width, label=label, yerr=errors, capsize=5)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('BERTScore Comparison Across Models', fontsize=14)
    plt.xticks(x + width, models, rotation=15, ha="right")
    plt.legend()
    plt.ylim(bottom=max(0, bertscore_df[metrics].min().min() - 0.05))
    plt.tight_layout()
    
    output_path_bertscore = Path(output_dir) / "bertscore_comparison.png"
    output_path_bertscore.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_bertscore, dpi=300)
    print(f"BERTScore plot saved successfully to: {output_path_bertscore}")
    plt.close()

    # --- 2. Plot BLEU and METEOR Comparison ---
    other_metrics_df = results_df.sort_values(by="meteor_score_mean", ascending=False)
    plt.figure(figsize=(12, 7))

    models = other_metrics_df['model'].tolist()
    x = np.arange(len(models))
    width = 0.35

    # Plot BLEU scores
    bleu_values = other_metrics_df['bleu_score'].tolist()
    plt.bar(x - width/2, bleu_values, width, label='BLEU Score')

    # Plot METEOR scores
    meteor_values = other_metrics_df['meteor_score_mean'].tolist()
    meteor_errors = other_metrics_df['meteor_score_std'].tolist()
    plt.bar(x + width/2, meteor_values, width, label='METEOR Score', yerr=meteor_errors, capsize=5)

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('BLEU and METEOR Score Comparison Across Models', fontsize=14)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.legend()
    all_values = other_metrics_df['bleu_score'].tolist() + other_metrics_df['meteor_score_mean'].tolist()
    plt.ylim(bottom=max(0, min(all_values) - 0.05))
    plt.tight_layout()

    output_path_other = Path(output_dir) / "other_metrics_comparison.png"
    plt.savefig(output_path_other, dpi=300)
    print(f"BLEU/METEOR plot saved successfully to: {output_path_other}")
    plt.close()

    # --- 3. Plot ROUGE Score Comparison ---
    rouge_df = results_df.sort_values(by="rougeL_fmeasure_mean", ascending=False)
    plt.figure(figsize=(12, 7))

    models = rouge_df['model'].tolist()
    metrics = ['rouge1_fmeasure_mean', 'rouge2_fmeasure_mean', 'rougeL_fmeasure_mean']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = rouge_df[metric].tolist()
        errors = rouge_df[metric.replace('mean', 'std')].tolist()
        plt.bar(x + i * width, values, width, label=label, yerr=errors, capsize=5)

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F-measure', fontsize=12)
    plt.title('ROUGE F-measure Comparison Across Models', fontsize=14)
    plt.xticks(x + width, models, rotation=15, ha="right")
    plt.legend()
    plt.ylim(bottom=max(0, rouge_df[metrics].min().min() - 0.05))
    plt.tight_layout()

    output_path_rouge = Path(output_dir) / "rouge_comparison.png"
    plt.savefig(output_path_rouge, dpi=300)
    print(f"ROUGE plot saved successfully to: {output_path_rouge}")
    plt.close()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Plot a comparison of summarization metrics from multiple CSV files."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/evaluation",
        help="Directory where the aggregated CSV result files are located."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/plots",
        help="Directory to save the final comparison plots."
    )
    args = parser.parse_args()
    
    plot_comparison_charts(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()