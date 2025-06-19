import json
import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

def ensure_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK 'wordnet' model...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading NLTK 'omw-1.4' model...")
        nltk.download('omw-1.4')


def save_results(results_data: dict, output_dir: str, model_name: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results to JSON
    detailed_filename = output_path / f"{model_name}_evaluation_detailed_{timestamp}.json"
    with open(detailed_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Detailed results saved to: {detailed_filename}")

    # Save aggregated results to CSV
    aggregated_data = {
        "model": [model_name],
        "rouge1_fmeasure_mean": [results_data.get("rouge1_fmeasure_mean")],
        "rouge1_fmeasure_std": [results_data.get("rouge1_fmeasure_std")],
        "rouge2_fmeasure_mean": [results_data.get("rouge2_fmeasure_mean")],
        "rouge2_fmeasure_std": [results_data.get("rouge2_fmeasure_std")],
        "rougeL_fmeasure_mean": [results_data.get("rougeL_fmeasure_mean")],
        "rougeL_fmeasure_std": [results_data.get("rougeL_fmeasure_std")],
        "bleu_score": [results_data.get("bleu_score")],
        "meteor_score_mean": [results_data.get("meteor_score_mean")],
        "meteor_score_std": [results_data.get("meteor_score_std")],
        "bert_score_precision_mean": [results_data["precision_mean"]],
        "bert_score_precision_std": [results_data["precision_std"]],
        "bert_score_recall_mean": [results_data["recall_mean"]],
        "bert_score_recall_std": [results_data["recall_std"]],
        "bert_score_f1_mean": [results_data["f1_mean"]],
        "bert_score_f1_std": [results_data["f1_std"]],
        "num_examples": [len(results_data["predictions"])]
    }
    df = pd.DataFrame(aggregated_data)
    aggregated_filename = output_path / f"{model_name}_evaluation_aggregated_{timestamp}.csv"
    df.to_csv(aggregated_filename, index=False)
    print(f"Aggregated results saved to: {aggregated_filename}")

def evaluate_summaries(file_path: str, model_type: str, output_dir: str):
    print(f"Loading summaries from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    if "summaries" not in data or "references" not in data:
        print("Error: The JSON file must contain 'summaries' and 'references' keys.")
        return

    generated_summaries = data["summaries"]
    reference_summaries = data["references"]

    ensure_nltk_data()

    # --- ROUGE Score Calculation ---
    print("Calculating ROUGE scores...")
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = scorer_rouge.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    results_data = {
        'rouge1_fmeasures': rouge1_scores,
        'rouge1_fmeasure_mean': np.mean(rouge1_scores),
        'rouge1_fmeasure_std': np.std(rouge1_scores),
        'rouge2_fmeasures': rouge2_scores,
        'rouge2_fmeasure_mean': np.mean(rouge2_scores),
        'rouge2_fmeasure_std': np.std(rouge2_scores),
        'rougeL_fmeasures': rougeL_scores,
        'rougeL_fmeasure_mean': np.mean(rougeL_scores),
        'rougeL_fmeasure_std': np.std(rougeL_scores),
    }

    # --- BLEU and METEOR Score Calculation ---
    print("Tokenizing summaries for BLEU and METEOR...")
    tokenized_generated = [word_tokenize(s.lower()) for s in generated_summaries]
    tokenized_references_bleu = [[word_tokenize(r.lower())] for r in reference_summaries]
    tokenized_references_meteor = [word_tokenize(r.lower()) for r in reference_summaries]

    print("Calculating BLEU score...")
    bleu_score_val = corpus_bleu(tokenized_references_bleu, tokenized_generated)
    results_data['bleu_score'] = bleu_score_val
    
    print("Calculating METEOR scores...")
    meteor_scores = [meteor_score([ref], gen) for ref, gen in zip(tokenized_references_meteor, tokenized_generated)]
    results_data.update({
        'meteor_scores': meteor_scores,
        'meteor_score_mean': np.mean(meteor_scores),
        'meteor_score_std': np.std(meteor_scores)
    })

    print(f"Found {len(generated_summaries)} summaries to evaluate.")
    print(f"Using BERTScore model: {model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing BERTScore on device: {device}")

    try:
        print(f"Manually loading tokenizer and model for '{model_type}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_type, trust_remote_code=True)
        model.to(device)
        print("Model and tokenizer loaded successfully.")

        scorer = BERTScorer(lang="en", rescale_with_baseline=False, device=device)
        
        scorer._model = model
        scorer._tokenizer = tokenizer
        print("BERTScorer instance updated with custom model.")
        
        (P, R, F1) = scorer.score(
            cands=generated_summaries,
            refs=reference_summaries,
            batch_size=8,
            verbose=True
        )

        results_data.update({
            "model_name": model_type,
            "predictions": generated_summaries,
            "references": reference_summaries,
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
            "precision_mean": P.mean().item(),
            "recall_mean": R.mean().item(),
            "f1_mean": F1.mean().item(),
            "precision_std": P.std().item(),
            "recall_std": R.std().item(),
            "f1_std": F1.std().item()
        })

        print("\n--- Evaluation Results ---")
        print(f"ROUGE-1 F-measure:   {results_data['rouge1_fmeasure_mean']:.4f}")
        print(f"ROUGE-2 F-measure:   {results_data['rouge2_fmeasure_mean']:.4f}")
        print(f"ROUGE-L F-measure:   {results_data['rougeL_fmeasure_mean']:.4f}")
        print(f"BLEU Score:          {results_data['bleu_score']:.4f}")
        print(f"Average METEOR:      {results_data['meteor_score_mean']:.4f}")
        print("--- BERTScore ---")
        print(f"Average Precision:   {results_data['precision_mean']:.4f}")
        print(f"Average Recall:      {results_data['recall_mean']:.4f}")
        print(f"Average F1 Score:    {results_data['f1_mean']:.4f}")
        print("--------------------------")

        save_results(results_data, output_dir, model_name=Path(file_path).stem.split('_')[0])

    except Exception as e:
        print(f"\nAn error occurred during score computation: {e}")
        print("Please ensure the model name is correct, you have an active internet connection, and all dependencies are installed.")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries using ROUGE, BERTScore, BLEU, and METEOR."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the JSON file containing generated and reference summaries."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Simonlee711/Clinical_ModernBERT",
        help="The Hugging Face model to use for BERTScore evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evaluation",
        help="Directory to save the evaluation results."
    )
    args = parser.parse_args()
    evaluate_summaries(args.file_path, args.model_type, args.output_dir)

if __name__ == "__main__":
    main()