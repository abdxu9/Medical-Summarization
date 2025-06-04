#!/usr/bin/env python3

"""
Medical Text Summarization Baseline Evaluation Script
====================================================

This script establishes baseline evaluations for abstractive summarization
of patient discharge documentation using Gemma 3-12B-IT and BART-Large-CNN.

Models supported:
- Gemma 3-12B-IT
- BART-Large-CNN

Dataset:
- MIMIC-IV-BHC (Brief Hospital Course summarization)

Evaluation metrics:
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import logging
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ML Libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig, Gemma3ForConditionalGeneration,
    set_seed
)
from huggingface_hub import login
from datasets import Dataset
import evaluate
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
login(token=os.environ.get('HUGGINGFACE_TOKEN'))

@dataclass
class ModelConfig:
    """Configuration for model and evaluation parameters"""
    name: str
    model_type: str  # "decoder" or "encoder-decoder"
    model_path: str
    max_length: int = 2048
    temperature: float = 0.7
    use_quantization: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for evaluation setup"""
    output_dir: str = "./results/baseline"
    dataset_path: str = "./data/mimic-iv-bhc.csv"
    sample_size: Optional[int] = 100  # Using 1000 examples as requested
    batch_size: int = 4
    max_new_tokens: int = 512
    random_seed: int = 42
    bert_score_model_type: str = "allenai/longformer-base-4096"
    save_examples: bool = True
    save_model_outputs: bool = True
    num_train: int = 80  # 80% of 1000
    num_val: int = 10    # 10% of 1000
    num_test: int = 10   # 10% of 1000

class MedicalSummarizationEvaluator:
    """Main class for running baseline evaluations"""
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_logging()
        self.setup_directories()
        self.load_evaluation_metrics()

        # Standard prompt for decoder models (Gemma)
        self.prompt_template = """You are a doctor in a hospital. You must summarize the patient's medical history, making sure to highlight the key elements so that our peers can quickly understand the situation, background, assessment, and recommendations regarding the patient.

Patient Record:

{input_text}

Summary:"""

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/baseline_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create output directories"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/summaries").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/metrics").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/visualizations").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def load_evaluation_metrics(self):
        """Initialize evaluation metrics"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore_evaluator = evaluate.load("bertscore")

    def get_model_configs(self) -> List[ModelConfig]:
        """Define model configurations for the requested models"""
        return [
            ModelConfig(
                name="gemma-3-12b-it",
                model_type="decoder",
                model_path="google/gemma-3-12b-it",
                max_length=4096  # Gemma 3 supports up to 128K tokens, but we limit for efficiency
            ),
            ModelConfig(
                name="bart-large-cnn",
                model_type="encoder-decoder",
                model_path="facebook/bart-large-cnn",
                max_length=1024
            )
        ]

    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the MIMIC-IV-BHC dataset into train, val, test"""
        self.logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        try:
            df = pd.read_csv(self.config.dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to load CSV file: {e}")
            raise
        
        # Check if the dataframe has the expected columns
        expected_columns = ['input', 'target']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"CSV file must contain {expected_columns} columns")
        
        # Rename columns for consistency
        df = df.rename(columns={'input': 'input_text', 'target': 'target_summary'})
        
        # Apply sample size limit if specified
        if self.config.sample_size and self.config.sample_size < len(df):
            
            df = df.sample(self.config.sample_size, random_state=self.config.random_seed)
            df = df.reset_index(drop=True)
        
        # Split into train, validation, and test sets (80/10/10)
        train_df = df.iloc[:self.config.num_train]
        val_df = df.iloc[self.config.num_train:self.config.num_train + self.config.num_val]
        test_df = df.iloc[self.config.num_train + self.config.num_val:self.config.num_train + self.config.num_val + self.config.num_test]
        
        # Convert pandas DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        self.logger.info(f"Dataset loaded and split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test examples")
        
        return train_dataset, val_dataset, test_dataset

    def setup_model(self, model_config: ModelConfig):
        """Setup model with quantization for memory efficiency"""
        self.logger.info(f"Setting up model: {model_config.name}")
        
        # Quantization configuration
        if model_config.use_quantization and model_config.model_type == "decoder":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.logger.info("Using 4-bit quantization")
        else:
            bnb_config = None
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        try:
            if model_config.model_type == "decoder":
                if "gemma-3" in model_config.model_path.lower():
                    # Special handling for Gemma 3 models
                    model = Gemma3ForConditionalGeneration.from_pretrained(
                        model_config.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_config.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16 if bnb_config else torch.float32,
                        trust_remote_code=True
                    )
            else:  # encoder-decoder
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        except Exception as e:
            self.logger.error(f"Failed to load model {model_config.name}: {e}")
            return None, None
            
        return model, tokenizer

    def generate_summaries(self, model, tokenizer, dataset: Dataset, model_config: ModelConfig) -> List[str]:
        """Generate summaries using the model"""
        self.logger.info(f"Generating summaries with {model_config.name}")
        
        summaries = []
        generation_times = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for i, example in enumerate(dataset):
            num_input_tokens = 0
            try:
                start_time = time.time()
                
                if model_config.model_type == "decoder":
                    # Format prompt for decoder models
                    prompt = self.prompt_template.format(input_text=example["input_text"])
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                      max_length=model_config.max_length).to(model.device)
                    
                    total_input_tokens += inputs.input_ids.shape[1]
                    
                    generation_kwargs = {
                        "max_new_tokens": self.config.max_new_tokens,
                        "do_sample": False,
                        "temperature": model_config.temperature,
                        "pad_token_id": tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id
                    }
                    num_input_tokens = inputs.input_ids.shape[1]
                    self.logger.info(f"(decoder model): Processed prompt tokens = {num_input_tokens} (tokenizer max_length: {model_config.max_length})")
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)
                    
                    # Extract only the generated part (excluding the prompt)
                    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                                     skip_special_tokens=True)
                    
                    total_output_tokens += len(outputs[0]) - inputs.input_ids.shape[1]
                    
                else:  # encoder-decoder
                    inputs = tokenizer(example["input_text"], return_tensors="pt", 
                                      truncation=True, max_length=model_config.max_length).to(model.device)
                    
                    total_input_tokens += inputs.input_ids.shape[1]
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            min_length=50,
                            length_penalty=2.0,
                            num_beams=4,
                            early_stopping=True,
                            temperature=model_config.temperature
                        )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    total_output_tokens += len(outputs[0])
                
                summaries.append(generated_text.strip())
                
                end_time = time.time()
                generation_time = end_time - start_time
                generation_times.append(generation_time)
                
                self.logger.info(f"Successfully generated summary {i+1}/{len(dataset)}. "
                                 f"Time for this summary: {generation_time:.2f}s. "
                                 f"Overall average time: {np.mean(generation_times):.2f}s")
                    
            except Exception as e:
                self.logger.error(f"Error generating summary for example {i}: {e}")
                summaries.append("")
                
                # If CUDA error, try to recover
                if "CUDA" in str(e):
                    self.logger.warning("CUDA error detected, attempting recovery...")
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                    except:
                        pass
        
        # Log statistics
        avg_time = np.mean(generation_times)
        avg_input_tokens = total_input_tokens / len(dataset)
        avg_output_tokens = total_output_tokens / len(dataset)
        
        self.logger.info(f"Summary generation completed for {model_config.name}")
        self.logger.info(f"Average generation time: {avg_time:.2f}s")
        self.logger.info(f"Average input tokens: {avg_input_tokens:.1f}")
        self.logger.info(f"Average output tokens: {avg_output_tokens:.1f}")
        
        return summaries

    def evaluate_summaries(self, generated_summaries: List[str], reference_summaries: List[str],
                          model_name: str) -> Dict:
        """Evaluate generated summaries using multiple metrics"""
        self.logger.info(f"Evaluating summaries for {model_name}")
        
        results = {
            "model": model_name,
            "num_examples": len(generated_summaries),
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
            "bert_score_precision": [],
            "bert_score_recall": [],
            "bert_score_f1": []
        }
        
        # ROUGE evaluation
        for gen, ref in zip(generated_summaries, reference_summaries):
            if not gen.strip():
                # Skip empty generations
                results["rouge_1"].append(0.0)
                results["rouge_2"].append(0.0)
                results["rouge_l"].append(0.0)
                continue
                
            rouge_scores = self.rouge_scorer.score(ref, gen)
            results["rouge_1"].append(rouge_scores['rouge1'].fmeasure)
            results["rouge_2"].append(rouge_scores['rouge2'].fmeasure)
            results["rouge_l"].append(rouge_scores['rougeL'].fmeasure)
            
        # BERTScore evaluation
        try:
            # Filter out empty summaries
            valid_pairs = [(gen, ref) for gen, ref in zip(generated_summaries, reference_summaries) 
                           if gen.strip()]
            
            if valid_pairs:
                valid_gen, valid_ref = zip(*valid_pairs)
                
                bs_results = self.bertscore_evaluator.compute(
                    predictions=list(valid_gen),
                    references=list(valid_ref),
                    model_type=self.config.bert_score_model_type,
                    lang="en",
                    batch_size=self.config.batch_size
                )
                
                # Fill in results for non-empty summaries
                valid_idx = 0
                for i in range(len(generated_summaries)):
                    if generated_summaries[i].strip():
                        results["bert_score_precision"].append(bs_results['precision'][valid_idx])
                        results["bert_score_recall"].append(bs_results['recall'][valid_idx])
                        results["bert_score_f1"].append(bs_results['f1'][valid_idx])
                        valid_idx += 1
                    else:
                        results["bert_score_precision"].append(0.0)
                        results["bert_score_recall"].append(0.0)
                        results["bert_score_f1"].append(0.0)
            else:
                # All summaries were empty
                results["bert_score_precision"] = [0.0] * len(generated_summaries)
                results["bert_score_recall"] = [0.0] * len(generated_summaries)
                results["bert_score_f1"] = [0.0] * len(generated_summaries)
                
        except Exception as e:
            self.logger.error(f"BERTScore evaluation failed: {e}")
            results["bert_score_precision"] = [0.0] * len(generated_summaries)
            results["bert_score_recall"] = [0.0] * len(generated_summaries)
            results["bert_score_f1"] = [0.0] * len(generated_summaries)
            
        # Aggregate metrics
        for metric in ["rouge_1", "rouge_2", "rouge_l", 
                      "bert_score_precision", "bert_score_recall", "bert_score_f1"]:
            results[f"{metric}_mean"] = np.mean(results[metric])
            results[f"{metric}_std"] = np.std(results[metric])
            
        return results

    def save_results(self, all_results: List[Dict], all_summaries: Dict):
        """Save evaluation results and generated summaries"""
        # Save aggregated results as CSV
        results_df = pd.DataFrame([{
            "model": r["model"],
            "rouge_1_mean": r["rouge_1_mean"],
            "rouge_1_std": r["rouge_1_std"],
            "rouge_2_mean": r["rouge_2_mean"],
            "rouge_2_std": r["rouge_2_std"],
            "rouge_l_mean": r["rouge_l_mean"],
            "rouge_l_std": r["rouge_l_std"],
            "bert_score_precision_mean": r["bert_score_precision_mean"],
            "bert_score_precision_std": r["bert_score_precision_std"],
            "bert_score_recall_mean": r["bert_score_recall_mean"],
            "bert_score_recall_std": r["bert_score_recall_std"],
            "bert_score_f1_mean": r["bert_score_f1_mean"],
            "bert_score_f1_std": r["bert_score_f1_std"],
            "num_examples": r["num_examples"]
        } for r in all_results])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.config.output_dir}/metrics/baseline_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Saved aggregated results to {results_file}")
        
        # Save detailed results as JSON
        detailed_file = f"{self.config.output_dir}/metrics/detailed_results_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        # Save generated summaries
        for model_name, summaries in all_summaries.items():
            summary_file = f"{self.config.output_dir}/summaries/{model_name}_summaries_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summaries, f, indent=2)
                
        self.logger.info("All results and summaries saved successfully")
        
        # Create visualizations
        self._create_visualizations(results_df, timestamp)
        
    def _create_visualizations(self, results_df: pd.DataFrame, timestamp: str):
        """Create visualizations of the results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set(style="whitegrid")
            
            # ROUGE scores comparison
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            models = results_df['model'].tolist()
            metrics = ['rouge_1_mean', 'rouge_2_mean', 'rouge_l_mean']
            metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            
            x = np.arange(len(models))
            width = 0.25
            
            # Plot bars
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = results_df[metric].tolist()
                errors = results_df[metric.replace('mean', 'std')].tolist()
                plt.bar(x + i*width, values, width, label=label, 
                       yerr=errors, capsize=5)
            
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.title('ROUGE Scores Comparison')
            plt.xticks(x + width, models)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.config.output_dir}/visualizations/rouge_comparison_{timestamp}.png")
            
            # BERTScore comparison
            plt.figure(figsize=(12, 6))
            
            metrics = ['bert_score_precision_mean', 'bert_score_recall_mean', 'bert_score_f1_mean']
            metric_labels = ['Precision', 'Recall', 'F1']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = results_df[metric].tolist()
                errors = results_df[metric.replace('mean', 'std')].tolist()
                plt.bar(x + i*width, values, width, label=label, 
                       yerr=errors, capsize=5)
            
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.title('BERTScore Comparison')
            plt.xticks(x + width, models)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.config.output_dir}/visualizations/bertscore_comparison_{timestamp}.png")
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")

    def run_baseline_evaluation(self):
        """Main method to run complete baseline evaluation"""
        self.logger.info("Starting baseline evaluation for medical text summarization")
        
        # Set random seed for reproducibility
        set_seed(self.config.random_seed)
        
        # Load and split dataset
        train_dataset, val_dataset, test_dataset = self.load_dataset()
        
        # We'll use the test set for evaluation
        dataset = test_dataset
        
        # Login to HuggingFace Hub if token is provided
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(hf_token)
        
        # Get model configurations
        model_configs = self.get_model_configs()
        
        all_results = []
        all_summaries = {}
        
        for model_config in model_configs:
            self.logger.info(f"Evaluating model: {model_config.name}")
            
            # Setup model
            model, tokenizer = self.setup_model(model_config)
            if model is None:
                self.logger.warning(f"Skipping {model_config.name} due to setup failure")
                continue
                
            # Generate summaries
            summaries = self.generate_summaries(model, tokenizer, dataset, model_config)
            all_summaries[model_config.name] = {
                "summaries": summaries,
                "references": dataset["target_summary"],
                "inputs": dataset["input_text"]
            }
            
            # Evaluate summaries
            results = self.evaluate_summaries(
                summaries,
                dataset["target_summary"],
                model_config.name
            )
            
            all_results.append(results)
            
            # Clean up GPU memory
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info(f"Completed evaluation for {model_config.name}")
            self.logger.info(f"ROUGE-L: {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}")
            self.logger.info(f"BERTScore F1: {results['bert_score_f1_mean']:.4f} ± {results['bert_score_f1_std']:.4f}")
            
        # Save all results
        self.save_results(all_results, all_summaries)
        
        self.logger.info("Baseline evaluation completed successfully")
        return all_results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Medical Text Summarization Baseline Evaluation")
    parser.add_argument("--output_dir", default="./results/baseline", help="Output directory for results")
    parser.add_argument("--dataset_path", default="./data/mimic-iv-bhc.csv", help="Path to the MIMIC-IV-BHC dataset")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of examples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bert_score_model", type=str, default="allenai/longformer-base-4096", 
                       help="Model type for BERTScore evaluation")
    
    args = parser.parse_args()
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        random_seed=args.seed,
        bert_score_model_type=args.bert_score_model
    )
    
    # Run evaluation
    evaluator = MedicalSummarizationEvaluator(eval_config)
    results = evaluator.run_baseline_evaluation()
    
    print("\n" + "="*50)
    print("BASELINE EVALUATION COMPLETED")
    print("="*50)
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"ROUGE-1: {result['rouge_1_mean']:.4f} ± {result['rouge_1_std']:.4f}")
        print(f"ROUGE-2: {result['rouge_2_mean']:.4f} ± {result['rouge_2_std']:.4f}")
        print(f"ROUGE-L: {result['rouge_l_mean']:.4f} ± {result['rouge_l_std']:.4f}")
        print(f"BERTScore F1: {result['bert_score_f1_mean']:.4f} ± {result['bert_score_f1_std']:.4f}")

if __name__ == "__main__":
    main()