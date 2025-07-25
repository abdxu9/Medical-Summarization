import os
import json
import pandas as pd
import torch
import logging
import warnings
import spacy
import nltk
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from bert_score import BERTScorer
from nltk.tokenize import sent_tokenize
import evaluate
from tqdm import tqdm
import gc

# --- Initial Setup ---
def setup_environment():
    """Configures logging, warnings, and checks for hardware acceleration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    warnings.filterwarnings("ignore", category=UserWarning, module='evaluate')
    warnings.filterwarnings("ignore", category=FutureWarning)

    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info(f"CUDA is available. Version: {torch.version.cuda}, Device count: {torch.cuda.device_count()}. Using device: {device}")
    else:
        device = 'cpu'
        logging.warning("CUDA not available. Running on CPU.")

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK's 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
    return device

# --- Core Functions ---

def clean_model_name(filename: str) -> str:
    """Cleans the JSON filename to derive a model name."""
    name = os.path.basename(filename).replace(".json", "")
    return name.replace("_summaries", "").replace("_summarization", "").replace("_", "-")

def load_data_from_json(results_dir: str = ".") -> dict:
    """Loads summarization data from JSON files in a specified directory."""
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        logging.error("No JSON files found in the directory.")
        return {}
    
    data_map = {}
    logging.info(f"Loading data from {len(json_files)} JSON files...")
    for file in tqdm(json_files, desc="Loading JSON files"):
        model_name = clean_model_name(file)
        file_path = os.path.join(results_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if all(k in data for k in ['summaries', 'references', 'inputs']) and \
               len(data['summaries']) == len(data['references']) == len(data['inputs']):
                data_map[model_name] = {
                    'predictions': data['summaries'],
                    'references': data['references'],
                    'inputs': data['inputs']
                }
            else:
                logging.warning(f"Skipping invalid file '{file}'. Check structure and key alignment.")
        except json.JSONDecodeError:
            logging.warning(f"Skipping malformed JSON file '{file}'.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading '{file}': {e}")
            
    logging.info("Data loading complete.")
    return data_map

def calculate_base_metrics(data_map: dict) -> pd.DataFrame:
    """Calculates ROUGE, BLEU, and METEOR scores for each model."""
    if not data_map: return pd.DataFrame()
    
    logging.info("Calculating Base Metrics (ROUGE, BLEU, METEOR)...")
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    
    all_results = []
    for model_name, data in tqdm(data_map.items(), desc="Calculating Base Metrics"):
        try:
            predictions = data['predictions']
            references = data['references']
            
            rouge_results = rouge.compute(predictions=predictions, references=references)
            bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
            meteor_results = meteor.compute(predictions=predictions, references=references)
            
            all_results.append({
                'Model': model_name,
                'ROUGE-1': rouge_results['rouge1'],
                'ROUGE-2': rouge_results['rouge2'],
                'ROUGE-L': rouge_results['rougeL'],
                'BLEU': bleu_results['bleu'],
                'METEOR': meteor_results['meteor']
            })
        except Exception as e:
            logging.error(f"Could not compute base metrics for {model_name}. Error: {e}")
            
    return pd.DataFrame(all_results)

def calculate_bertscore(data_map: dict, device: str) -> pd.DataFrame:
    """
    Calculates BERTScore for multiple models using manual loading for memory optimization.
    """
    if not data_map: return pd.DataFrame()

    bert_models = {
        # All models should now load correctly with this method
        "Clinical-ModernBERT": "Simonlee711/Clinical_ModernBERT",
        "ModernBERT-base": "answerdotai/ModernBERT-base",
        "Longformer-base": "allenai/longformer-base-4096"
    }
    
    all_results = []
    logging.info("--- Calculating BERTScore with Memory Optimization ---")

    for bert_name, bert_path in bert_models.items():
        logging.info(f"Initializing BERTScore model: {bert_name} ({bert_path})")
        
        try:
            # --- Applying your successful loading strategy ---
            # 1. Load tokenizer with specific logic for model types
            use_fast_tokenizer = "longformer" not in bert_path.lower()
            
            tokenizer = AutoTokenizer.from_pretrained(
                bert_path, 
                trust_remote_code=True, 
                use_fast=use_fast_tokenizer
            )
            
            # 2. Load model in half-precision to save memory
            model = AutoModel.from_pretrained(
                bert_path, 
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager for stability
                torch_dtype=torch.float16      # The key memory optimization
            )
            model.eval()
            model = model.to(device)
            logging.info(f"Successfully loaded {bert_name} in half-precision (float16).")
            
            # 3. Manually inject the optimized model into BERTScorer
            # Note: rescale_with_baseline=False is crucial for memory savings
            scorer = BERTScorer(lang="en", rescale_with_baseline=False, device=device)
            scorer._model = model
            scorer._tokenizer = tokenizer
            
            # --- Scoring Loop ---
            for model_name, data in tqdm(data_map.items(), desc=f"Scoring with {bert_name}"):
                try:
                    preds, refs = data['predictions'], data['references']
                    if not preds or not refs:
                        F1 = torch.tensor([0.0]) # Handle empty predictions
                    else:
                        # Use a small, safe batch size
                        _, _, F1 = scorer.score(cands=preds, refs=refs, batch_size=2, verbose=False)

                    all_results.append({
                        'Model': model_name, 
                        'BERTScore_Model': bert_name,
                        'BERTScore-F1': F1.mean().item()
                    })
                except Exception as e:
                    logging.error(f"Could not score {model_name} with {bert_name}. Error: {e}", exc_info=True)
                    all_results.append({'Model': model_name, 'BERTScore_Model': bert_name, 'BERTScore-F1': "N/A"})
            
            # Clean up memory before loading the next model
            del model, tokenizer, scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logging.error(f"Could not load or initialize BERTScore model {bert_name}. Skipping. Error: {e}", exc_info=True)
            for model_name in data_map.keys():
                all_results.append({'Model': model_name, 'BERTScore_Model': bert_name, 'BERTScore-F1': "N/A"})
                
    # --- Reformat the DataFrame ---
    # We only have F1 score now, so let's pivot that
    if not all_results: return pd.DataFrame()
    
    temp_df = pd.DataFrame(all_results)
    bertscore_pivot = temp_df.pivot(
        index='Model', 
        columns='BERTScore_Model', 
        values='BERTScore-F1'
    )
    # Add prefix to column names for clarity
    bertscore_pivot = bertscore_pivot.add_prefix('BERTScore-F1_')
    bertscore_pivot.reset_index(inplace=True)
    
    return bertscore_pivot

def check_entailment_and_contradiction(summary: str, source_text: str, nli_model: AutoModelForSequenceClassification, nli_tokenizer: AutoTokenizer, device: str) -> dict:
    """Analyzes a single summary for entailment/contradiction against a source text."""
    sentences = sent_tokenize(summary)
    results = {"entailment": 0, "contradiction": 0, "neutral": 0, "total_sentences": len(sentences)}
    if not sentences: return results

    for sentence in sentences:
        try:
            inputs = nli_tokenizer(source_text, sentence, return_tensors="pt", truncation=True, max_length=nli_tokenizer.model_max_length).to(device)
            with torch.no_grad():
                outputs = nli_model(**inputs)
            
            scores = torch.softmax(outputs.logits, dim=-1)[0]
            label_id = torch.argmax(scores).item()
            label = nli_model.config.id2label[label_id]

            if label in results:
                results[label] += 1
        except Exception as e:
            logging.error(f"Could not perform NLI on sentence: '{sentence}'. Error: {e}")
    return results

### NEW FUNCTION ###
def calculate_nli_metrics(data_map: dict, device: str) -> pd.DataFrame:
    """
    Performs NLI analysis across the entire dataset for each model.
    WARNING: This is computationally very expensive.
    """
    logging.warning("Starting full NLI analysis. This process can take a very long time.")
    
    # NLI Model Setup
    nli_model, nli_tokenizer = None, None
    try:
        nli_model_name = "cross-encoder/nli-deberta-v3-base"
        logging.info(f"Loading NLI model ({nli_model_name}) on device: {device}...")
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    except Exception as e:
        logging.error(f"Could not load NLI model. NLI metrics will not be calculated. Error: {e}")
        return pd.DataFrame()

    all_nli_results = []

    for model_name, data in data_map.items():
        total_entailment = 0
        total_contradiction = 0
        total_sentences = 0
        
        # Use tqdm for progress tracking on this long-running task
        progress_bar = tqdm(
            zip(data['predictions'], data['inputs']), 
            total=len(data['predictions']),
            desc=f"NLI for {model_name}"
        )
        
        for pred, source in progress_bar:
            if not pred or not source:
                continue
                
            nli_result = check_entailment_and_contradiction(pred, source, nli_model, nli_tokenizer, device)
            total_entailment += nli_result['entailment']
            total_contradiction += nli_result['contradiction']
            total_sentences += nli_result['total_sentences']

        # Calculate average rates for the model
        avg_entailment_rate = (total_entailment / total_sentences) if total_sentences > 0 else 0
        avg_contradiction_rate = (total_contradiction / total_sentences) if total_sentences > 0 else 0
        
        all_nli_results.append({
            'Model': model_name,
            'Entailment-Rate': avg_entailment_rate,
            'Contradiction-Rate': avg_contradiction_rate
        })
        
        logging.info(f"Finished NLI for {model_name}. Entailment: {avg_entailment_rate:.2%}, Contradiction: {avg_contradiction_rate:.2%}")

    return pd.DataFrame(all_nli_results)

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Setup Environment
    selected_device = setup_environment()
    
    # 2. Load Data
    data = load_data_from_json()
    
    if data:
        # 3. Run Quantitative Metrics
        # NOTE: The qualitative 'spot-check' function has been replaced by the full quantitative calculation below.
        base_metrics_df = calculate_base_metrics(data)
        bertscore_df = calculate_bertscore(data, selected_device)
        
        # 4. Calculate NLI metrics across the full dataset
        nli_metrics_df = calculate_nli_metrics(data, selected_device)
        
        # 5. Combine and Save Results
        if not base_metrics_df.empty:
            full_results_df = base_metrics_df.copy()
            
            # Corrected block: Directly merge the pre-pivoted DataFrame
            if not bertscore_df.empty:
                full_results_df = pd.merge(full_results_df, bertscore_df, on="Model", how="left")
            
            # Merge NLI metrics
            if not nli_metrics_df.empty:
                full_results_df = pd.merge(full_results_df, nli_metrics_df, on="Model", how="left")

            print("\n\n--- Combined Metrics ---")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(full_results_df.to_string())
            
            output_csv_path = "full_analysis_results.csv"
            full_results_df.to_csv(output_csv_path, index=False)
            logging.info(f"\nResults saved to {output_csv_path}")