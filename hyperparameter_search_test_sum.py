#!/usr/bin/env python3

"""
Hyperparameter Optimization for Medical Text Summarization
=========================================================

This script performs hyperparameter optimization for fine-tuning Gemma 3-12B-IT and
BART-Large-CNN on medical text summarization using QLoRA and LoRA, respectively.

Models supported:
- Gemma 3-12B-IT (with QLoRA)
- BART-Large-CNN (with LoRA)

Dataset:
- MIMIC-IV-BHC (Brief Hospital Course summarization)

Optimization method:
- Random search across stratified hyperparameter spaces
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import pandas as pd
import numpy as np
import torch
import logging
import argparse
import gc
import time
import random
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ML Libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
    Gemma3ForConditionalGeneration, set_seed
)
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, PeftConfig
from datasets import Dataset
from trl import SFTTrainer
import evaluate
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
login(token=os.environ.get('HUGGINGFACE_TOKEN'))

@dataclass
class HyperparameterSpace:
    """Define the hyperparameter search space"""
    lora_rank_min: int = 4
    lora_rank_max: int = 64
    lora_alpha_min: int = 8
    lora_alpha_max: int = 128
    lora_dropout_min: float = 0.05
    lora_dropout_max: float = 0.3
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 5e-4
    batch_size_choices: List[int] = field(default_factory=lambda: [1, 2, 4])
    warmup_ratio_min: float = 0.03
    warmup_ratio_max: float = 0.1
    
@dataclass
class ModelConfig:
    """Configuration for model and training parameters"""
    name: str
    model_type: str  # "decoder" or "encoder-decoder"
    model_path: str
    lora_target_modules: Optional[List[str]] = None
    max_length: int = 2048
    temperature: float = 0.7
    use_quantization: bool = True
    
    def __post_init__(self):
        """Set default target modules based on model type"""
        if self.lora_target_modules is None:
            if self.model_type == "decoder":
                # Default for decoder models like Gemma
                self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default for encoder-decoder models like BART
                self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    
@dataclass
class SearchConfig:
    """Configuration for hyperparameter search"""
    method: str = "random"  # "random" search as specified
    n_trials: int = 1      # Number of trials to run
    output_dir: str = "./results/hyperparameter"
    dataset_path: str = "./data/mimic-iv-bhc.csv"
    sample_size: Optional[int] = 100  # Using 1000 examples as requested
    eval_steps: int = 50
    max_steps: int = 10    # Limit for each trial
    save_strategy: str = "steps"
    gradient_checkpointing: bool = True
    max_memory_per_gpu: Optional[str] = "30GiB"  # For RTX 5090
    random_seed: int = 42
    num_train: int = 80  # 80% of 1000
    num_val: int = 10    # 10% of 1000
    num_test: int = 10   # 10% of 1000

class MedicalSummarizationOptimizer:
    """Main class for hyperparameter optimization"""
    def __init__(self, hp_space: HyperparameterSpace, search_config: SearchConfig):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.hp_space = hp_space
        self.config = search_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_logging()
        self.setup_directories()
        
        # Prepare dataset
        self.train_dataset, self.eval_dataset, self.test_dataset = self.prepare_dataset()
        
        # For storing the best results
        self.best_results = {}

        # Define the prompt template as a class attribute
        self.prompt_template = """You are a doctor in a hospital. You must summarize the patient's medical history, making sure to highlight the key elements so that our peers can quickly understand the situation, background, assessment, and recommendations regarding the patient.

Patient Record:

{input_text}

Summary:"""
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/hyperparameter_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create output directories"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/models").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/results").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/visualizations").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def prepare_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare training, validation, and test datasets"""
        self.logger.info("Preparing datasets for hyperparameter optimization")
        
        # Load actual dataset
        try:
            df = pd.read_csv(self.config.dataset_path)
            
            if 'input' not in df.columns or 'target' not in df.columns:
                raise ValueError("CSV must contain 'input' and 'target' columns")
                
            df = df.rename(columns={'input': 'input_text', 'target': 'target_summary'})
            df_processed = df[['input_text', 'target_summary']].dropna()
            
            if self.config.sample_size and self.config.sample_size < len(df_processed):
                df_processed = df_processed.sample(self.config.sample_size, random_state=self.config.random_seed)
                df_processed = df_processed.reset_index(drop=True)
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
            
        # Split into train, validation, and test
        train_df = df_processed.iloc[:self.config.num_train]
        val_df = df_processed.iloc[self.config.num_train:self.config.num_train + self.config.num_val]
        test_df = df_processed.iloc[self.config.num_train + self.config.num_val:self.config.num_train + self.config.num_val + self.config.num_test]
        
        # Convert to HF datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        self.logger.info(f"Datasets prepared: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_model_configs(self) -> List[ModelConfig]:
        """Define model configurations"""
        return [
            ModelConfig(
                name="gemma-3-12b-it",
                model_type="decoder",
                model_path="google/gemma-3-12b-it",
                max_length=8192,  # Limiting context for efficiency
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            ModelConfig(
                name="bart-large-cnn",
                model_type="encoder-decoder",
                model_path="facebook/bart-large-cnn",
                max_length=1024,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            )
        ]
        
    def create_model_with_lora(self, model_config: ModelConfig, trial_params: Dict[str, Any]):
        """Create model with LoRA configuration based on trial parameters"""
        self.logger.info(f"Creating {model_config.name} with LoRA (rank={trial_params['lora_rank']})")
        
        # Quantization config optimized for RTX 5090
        if model_config.use_quantization and model_config.model_type == "decoder":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.logger.info("Using 4-bit quantization")
        else:
            bnb_config = None
            
        # Load base model with memory optimization
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            # "low_cpu_mem_usage": True, # Removed, as baseline.py doesn't use it for BART and works
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
            
        # Try removing explicit max_memory to let accelerate handle it with device_map="auto"
        # if self.config.max_memory_per_gpu:
        #     model_kwargs["max_memory"] = {0: self.config.max_memory_per_gpu}
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model based on type
        if model_config.model_type == "encoder-decoder":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.model_path,
                **model_kwargs
            )
            task_type = TaskType.SEQ_2_SEQ_LM
        elif model_config.model_type == "decoder":
            if "gemma-3" in model_config.model_path.lower():
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_config.model_path,
                    **model_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_path,
                    **model_kwargs
                )
            task_type = TaskType.CAUSAL_LM
        else:
            self.logger.error(f"Unsupported model_type: {model_config.model_type}")
            raise ValueError(f"Unsupported model_type: {model_config.model_type}")
            
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=task_type,
            r=trial_params["lora_rank"],
            lora_alpha=trial_params["lora_alpha"],
            lora_dropout=trial_params["lora_dropout"],
            target_modules=model_config.lora_target_modules
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Disable cache for training compatibility with gradient checkpointing
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            
        # Enable gradients for LoRA parameters
        model.enable_input_require_grads()
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return model, tokenizer
        
    def tokenize_dataset(self, dataset: Dataset, tokenizer, model_config: ModelConfig) -> Dataset:
        """Tokenize dataset based on model type"""
        self.logger.info(f"Tokenizing dataset for {model_config.name}")
        
        prompt_template = """You are a doctor in a hospital. You must summarize the patient's medical history, making sure to highlight the key elements so that our peers can quickly understand the situation, background, assessment, and recommendations regarding the patient.

Patient Record:

{input_text}

Summary:"""
        
        def tokenize_function(examples):
            if model_config.model_type == "decoder":
                # Format each example with prompt
                formatted_inputs = [prompt_template.format(input_text=text) for text in examples["input_text"]]
                
                # Tokenize inputs
                tokenized_inputs = tokenizer(
                    formatted_inputs,
                    padding="max_length", 
                    truncation=True,
                    max_length=model_config.max_length,
                    return_tensors="pt"
                )
                
                # Tokenize labels (target summaries)
                tokenized_outputs = tokenizer(
                    examples["target_summary"],
                    padding="max_length",
                    truncation=True,
                    max_length=512,  # Limit summary length
                    return_tensors="pt"
                )
                
                # Create label tensors with -100 for prompt tokens (to ignore them in loss calculation)
                labels = copy.deepcopy(tokenized_inputs["input_ids"])
                for i, (input_text, target) in enumerate(zip(formatted_inputs, examples["target_summary"])):
                    # Tokenize just the prompt to find its length
                    prompt_tokens = tokenizer(prompt_template.format(input_text=""), add_special_tokens=False)["input_ids"]
                    prompt_len = len(prompt_tokens) + len(tokenizer(examples["input_text"][i], add_special_tokens=False)["input_ids"])
                    
                    # Set prompt tokens to -100 (ignore in loss)
                    labels[i, :prompt_len] = -100
                    
                    # Add tokenized target (truncated to fit)
                    target_tokens = tokenized_outputs["input_ids"][i]
                    target_len = min(len(target_tokens), model_config.max_length - prompt_len)
                    if prompt_len < model_config.max_length:
                        labels[i, prompt_len:prompt_len+target_len] = target_tokens[:target_len]
                    
                    # Set padding tokens to -100
                    padding_mask = tokenized_inputs["attention_mask"][i] == 0
                    labels[i, padding_mask] = -100
                
                return {
                    "input_ids": tokenized_inputs["input_ids"],
                    "attention_mask": tokenized_inputs["attention_mask"],
                    "labels": labels
                }
            else:  # encoder-decoder
                # Tokenize inputs
                tokenized_inputs = tokenizer(
                    examples["input_text"],
                    padding="max_length",
                    truncation=True,
                    max_length=model_config.max_length,
                    return_tensors="pt"
                )
                
                # Tokenize outputs
                tokenized_outputs = tokenizer(
                    examples["target_summary"],
                    padding="max_length",
                    truncation=True,
                    max_length=512,  # Limit summary length
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": tokenized_inputs["input_ids"],
                    "attention_mask": tokenized_inputs["attention_mask"],
                    "labels": tokenized_outputs["input_ids"]
                }
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["input_text", "target_summary"]
        )
        
        return tokenized_dataset
    
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample hyperparameters from the defined space"""
        params = {
            "lora_rank": random.randint(self.hp_space.lora_rank_min, self.hp_space.lora_rank_max),
            "lora_alpha": random.randint(self.hp_space.lora_alpha_min, self.hp_space.lora_alpha_max),
            "lora_dropout": random.uniform(self.hp_space.lora_dropout_min, self.hp_space.lora_dropout_max),
            "learning_rate": random.uniform(self.hp_space.learning_rate_min, self.hp_space.learning_rate_max),
            "batch_size": random.choice(self.hp_space.batch_size_choices),
            "warmup_ratio": random.uniform(self.hp_space.warmup_ratio_min, self.hp_space.warmup_ratio_max)
        }
        return params
        
    def evaluate_trial(self, model_config: ModelConfig, trial_params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Train and evaluate a model with the given hyperparameters"""
        trial_id = f"{model_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting trial {trial_id} with params: {trial_params}")
        
        # Create model with LoRA
        model, tokenizer = self.create_model_with_lora(model_config, trial_params)
        
        # Tokenize datasets
        train_dataset_tokenized = self.tokenize_dataset(self.train_dataset, tokenizer, model_config)
        eval_dataset_tokenized = self.tokenize_dataset(self.eval_dataset, tokenizer, model_config)
        
        # Data collator
        if model_config.model_type == "decoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        else:  # encoder-decoder
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8
            )
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/models/{trial_id}",
            num_train_epochs=3,
            per_device_train_batch_size=trial_params["batch_size"],
            per_device_eval_batch_size=trial_params["batch_size"],
            gradient_accumulation_steps=max(1, 8 // trial_params["batch_size"]),  # Effective batch size of 8
            learning_rate=trial_params["learning_rate"],
            warmup_ratio=trial_params["warmup_ratio"],
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.eval_steps,
            max_steps=self.config.max_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            tf32=True,
            group_by_length=True,
            optim="adamw_torch_fused"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=eval_dataset_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Train model
        try:
            trainer.train()
            
            # Get final evaluation loss
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss", float('inf'))
            
            # Evaluate on test set
            test_dataset_tokenized = self.tokenize_dataset(self.test_dataset, tokenizer, model_config)
            test_results = trainer.evaluate(test_dataset_tokenized)
            test_loss = test_results.get("eval_loss", float('inf'))
            
            # Generate some summaries for ROUGE and BERTScore evaluation
            
            # rouge_evaluator is self.rouge_evaluator, bertscore_evaluator is self.bertscore_evaluator
            # from __init__ if you refactor metric loading there, or define them here.
            rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Load BERTScore with a smaller model to save memory
            self.logger.info("Loading BERTScore evaluator with roberta-base model...")
            try:
                bertscore_evaluator = evaluate.load("bertscore", model_type="roberta-base")
            except Exception as e_bert_load:
                self.logger.error(f"Failed to load BERTScore with roberta-base: {e_bert_load}. Falling back to default.")
                bertscore_evaluator = evaluate.load("bertscore") # Fallback to default if roberta-base fails for some reason

            # Sample a subset for metric evaluation
            sample_size = min(20, len(self.test_dataset)) # Using 20 examples for ROUGE/BERTScore
            test_sample = self.test_dataset.select(range(sample_size))
            
            generated_summaries_for_test_sample = []
            reference_summaries_for_test_sample = [ex["target_summary"] for ex in test_sample]
            per_example_rouge_scores = [] # To store dicts of ROUGE scores

            # Use the class-level prompt_template
            prompt_template_to_use = self.prompt_template 

            self.logger.info(f"Generating {len(test_sample)} summaries for ROUGE and BERTScore evaluation...")
            for i, example in enumerate(test_sample):
                current_generated_text = ""
                try:
                    if model_config.model_type == "decoder":
                        prompt = prompt_template_to_use.format(input_text=example["input_text"])
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                          max_length=model_config.max_length).to(model.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs, 
                                max_new_tokens=512, # Consider making this configurable
                                do_sample=False,    # Consistent with original, no sampling
                                temperature=model_config.temperature 
                            )
                        current_generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    else:  # encoder-decoder
                        inputs = tokenizer(example["input_text"], return_tensors="pt", truncation=True,
                                          max_length=model_config.max_length).to(model.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512, # Consider making this configurable
                                do_sample=False     # Consistent with original
                            )
                        current_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    current_generated_text = current_generated_text.strip()
                except Exception as e_gen:
                    self.logger.error(f"Error generating summary for example {i}: {e_gen}")
                    current_generated_text = "" # Fallback to empty string

                generated_summaries_for_test_sample.append(current_generated_text)
                
                # Calculate ROUGE for this example
                rouge_result = rouge_evaluator.score(example["target_summary"], current_generated_text)
                per_example_rouge_scores.append({
                    "rouge1": rouge_result["rouge1"].fmeasure,
                    "rouge2": rouge_result["rouge2"].fmeasure,
                    "rougeL": rouge_result["rougeL"].fmeasure
                })

            self.logger.info("Summary generation for metrics complete.")

            # --- Save model BEFORE deleting trainer, if it's the best one ---
            current_eval_loss_for_best_model_check = eval_results.get("eval_loss", float('inf')) # Use eval_loss from validation
            is_best_so_far = False
            if model_config.name not in self.best_results or \
               current_eval_loss_for_best_model_check < self.best_results[model_config.name]["eval_loss"]:
                is_best_so_far = True
            
            if is_best_so_far:
                best_model_path = f"{self.config.output_dir}/models/best_{model_config.name}"
                Path(best_model_path).mkdir(parents=True, exist_ok=True)
                if 'trainer' in locals() and trainer is not None:
                    self.logger.info(f"Saving new best model for {model_config.name} to {best_model_path} (eval_loss: {current_eval_loss_for_best_model_check:.4f})")
                    trainer.save_model(best_model_path)
                else:
                    self.logger.warning("Trainer object not available for saving the best model. This should not happen.")

            # Now that all generations and potential model saving are done, release the main model and trainer from GPU
            self.logger.info("Releasing main model and trainer from GPU before BERTScore calculation.")
            if 'trainer' in locals() and trainer is not None:
                del trainer
                self.logger.info("Trainer object deleted.")
            if 'model' in locals() and model is not None: # model is part of trainer, but also a direct reference
                del model
                self.logger.info("Model object deleted.")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("GPU memory cleared for BERTScore.")
            
            # Calculate BERTScore
            # Ensure predictions and references are lists of strings
            # Filter out empty predictions for BERTScore if the library handles it poorly,
            # or ensure it can handle them. The original script didn't explicitly filter.
            # For now, pass all generated summaries.
            self.logger.info(f"Calculating BERTScore for {len(generated_summaries_for_test_sample)} samples...")
            # Use a smaller batch_size for BERTScore computation
            bert_score_calculation_batch_size = 4 
            bert_results = bertscore_evaluator.compute(
                predictions=generated_summaries_for_test_sample,
                references=reference_summaries_for_test_sample,
                lang="en",
                batch_size=bert_score_calculation_batch_size
            )
            self.logger.info("BERTScore calculation complete.")
            
            # Aggregate ROUGE scores
            avg_rouge1 = np.mean([score["rouge1"] for score in per_example_rouge_scores])
            avg_rouge2 = np.mean([score["rouge2"] for score in per_example_rouge_scores])
            avg_rougeL = np.mean([score["rougeL"] for score in per_example_rouge_scores])
            
            # Average BERTScore (ensure 'f1' is a list/array of numbers)
            avg_bert_f1 = np.mean(bert_results["f1"]) if "f1" in bert_results and len(bert_results["f1"]) > 0 else 0.0
            
            # Combine all metrics
            # eval_loss and test_loss come from trainer.evaluate() calls earlier in the try block
            metrics = {
                "eval_loss": eval_results.get("eval_loss", float('inf')), # Assuming eval_results is populated
                "test_loss": test_results.get("eval_loss", float('inf')), # Assuming test_results is populated
                "rouge1": avg_rouge1,
                "rouge2": avg_rouge2,
                "rougeL": avg_rougeL,
                "bertscore_f1": avg_bert_f1
            }
            
            self.logger.info(f"Trial {trial_id} completed with eval_loss={metrics['eval_loss']:.4f}, test_loss={metrics['test_loss']:.4f}")
            self.logger.info(f"ROUGE-1={metrics['rouge1']:.4f}, ROUGE-2={metrics['rouge2']:.4f}, ROUGE-L={metrics['rougeL']:.4f}, BERTScore F1={metrics['bertscore_f1']:.4f}")
            
            # Update self.best_results dictionary AFTER all metrics are computed, using the is_best_so_far flag
            if is_best_so_far: # is_best_so_far was determined before deleting trainer
                self.best_results[model_config.name] = {
                    "trial_id": trial_id,
                    "params": trial_params,
                    "eval_loss": current_eval_loss_for_best_model_check, 
                    "metrics": metrics # The full metrics dict
                }
                self.logger.info(f"Updated best results for {model_config.name}.")
            
            return metrics['eval_loss'], metrics # Return the eval_loss from the metrics dict for consistency
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial_id}: {e}")
            return float('inf'), {"error": str(e)}
        finally:
            # Clean up safely, variables might have been deleted already
            self.logger.info("Entering finally block for cleanup in evaluate_trial.")
            try:
                if 'model' in locals() and model is not None:
                    del model
                    self.logger.info("Model deleted in finally.")
            except NameError:
                self.logger.info("Model was not defined or already deleted before finally.")
            
            try:
                if 'tokenizer' in locals() and tokenizer is not None: # tokenizer is not deleted before, so this should be fine
                    del tokenizer
                    self.logger.info("Tokenizer deleted in finally.")
            except NameError:
                self.logger.info("Tokenizer was not defined before finally.")

            try:
                if 'trainer' in locals() and trainer is not None:
                    del trainer
                    self.logger.info("Trainer deleted in finally.")
            except NameError:
                self.logger.info("Trainer was not defined or already deleted before finally.")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Final GPU cache clear in evaluate_trial.")
            
    def random_search(self, model_config: ModelConfig):
        """Perform random search hyperparameter optimization"""
        self.logger.info(f"Starting random search for {model_config.name}")
        
        results = []
        best_trial = None
        best_value = float('inf')
        
        for trial_idx in range(self.config.n_trials):
            # Sample hyperparameters
            trial_params = self.sample_hyperparameters()
            self.logger.info(f"Trial {trial_idx+1}/{self.config.n_trials} for {model_config.name}")
            
            # Evaluate trial
            eval_loss, metrics = self.evaluate_trial(model_config, trial_params)
            
            # Store results
            result = {
                "trial_number": trial_idx,
                "params": trial_params,
                "eval_loss": eval_loss,
                "metrics": metrics
            }
            results.append(result)
            
            # Update best trial
            if eval_loss < best_value:
                best_value = eval_loss
                best_trial = result
                self.logger.info(f"New best trial found: {trial_idx} with eval_loss={eval_loss:.4f}")
                
        # Print results
        self.logger.info(f"Random search completed for {model_config.name}")
        if best_trial:
            self.logger.info(f"Best trial: {best_trial['trial_number']} with eval_loss={best_trial['eval_loss']:.4f}")
            self.logger.info(f"Best parameters: {best_trial['params']}")
            self.logger.info(f"Best metrics: {best_trial['metrics']}")
            
        return {"trials": results, "best_trial": best_trial}
    
    def save_hyperparameter_search_details(self, all_hyperparam_results: Dict[str, Dict]):
        """Saves the detailed results of all hyperparameter search trials."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.search_timestamp = timestamp # Store for use in other saving functions

        results_file = f"{self.config.output_dir}/results/hyperparameter_search_details_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_hyperparam_results, f, indent=2)
        self.logger.info(f"Detailed hyperparameter search results saved to {results_file}")

        # Visualizations for hyperparameter search
        self._create_visualizations(all_hyperparam_results, timestamp)

    def generate_and_save_best_model_outputs(self):
        """Loads the best fine-tuned model for each type, generates summaries on the test set, and saves them."""
        self.logger.info("Generating summaries from best models identified during hyperparameter search...")

        if not hasattr(self, 'search_timestamp'):
            self.search_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, best_trial_data in self.best_results.items():
            if not best_trial_data:
                self.logger.warning(f"No best trial data found for {model_name}. Skipping summary generation.")
                continue

            self.logger.info(f"Processing best model for: {model_name}")

            current_model_config = next((mc for mc in self.get_model_configs() if mc.name == model_name), None)
            if not current_model_config:
                self.logger.error(f"Could not find ModelConfig for {model_name}. Skipping.")
                continue

            best_adapter_path = f"{self.config.output_dir}/models/best_{model_name}"
            if not Path(best_adapter_path).exists():
                self.logger.warning(f"Best model adapter path not found for {model_name} at {best_adapter_path}. Skipping summary generation.")
                continue

            self.logger.info(f"Loading best PEFT model for {model_name} from {best_adapter_path}")

            bnb_config_final = None
            if current_model_config.use_quantization and current_model_config.model_type == "decoder":
                bnb_config_final = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )

            model_kwargs_final = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if bnb_config_final:
                model_kwargs_final["quantization_config"] = bnb_config_final
                model_kwargs_final["torch_dtype"] = torch.bfloat16 if "gemma" in current_model_config.model_path.lower() else torch.float16
            else:
                model_kwargs_final["torch_dtype"] = torch.float16

            tokenizer_final = AutoTokenizer.from_pretrained(current_model_config.model_path)
            if tokenizer_final.pad_token is None:
                tokenizer_final.pad_token = tokenizer_final.eos_token

            base_model_final = None
            try:
                if current_model_config.model_type == "decoder":
                    if "gemma-3" in current_model_config.model_path.lower():
                        base_model_final = Gemma3ForConditionalGeneration.from_pretrained(
                            current_model_config.model_path, **model_kwargs_final
                        )
                    else:
                        base_model_final = AutoModelForCausalLM.from_pretrained(
                            current_model_config.model_path, **model_kwargs_final
                        )
                else:
                    base_model_final = AutoModelForSeq2SeqLM.from_pretrained(
                        current_model_config.model_path, **model_kwargs_final
                    )
            except Exception as e:
                self.logger.error(f"Failed to load base model {current_model_config.name} for final summarization: {e}")
                continue

            model_final = None
            try:
                model_final = PeftModel.from_pretrained(base_model_final, best_adapter_path)
                model_final.eval()
                self.logger.info(f"Successfully loaded PEFT model for {model_name}.")
            except Exception as e:
                self.logger.error(f"Failed to load PEFT adapter for {model_name} from {best_adapter_path}: {e}")
                del base_model_final, tokenizer_final
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            generated_summaries_final = []
            generation_times_final = []
            
            # Use a SearchConfig like structure for generation parameters
            # Let's use a simple dictionary for generation params for now or adapt self.config
            # For this specific task, we'll define max_new_tokens here.
            # Max new tokens for the best model summarization output
            max_new_tokens_for_final_summary = 512 


            self.logger.info(f"Generating summaries for {model_name} on the test set ({len(self.test_dataset)} examples)...")
            for i, example in enumerate(self.test_dataset):
                try:
                    start_time_final = time.time()
                    if current_model_config.model_type == "decoder":
                        prompt = self.prompt_template.format(input_text=example["input_text"])
                        inputs = tokenizer_final(prompt, return_tensors="pt", truncation=True,
                                               max_length=current_model_config.max_length).to(model_final.device)
                        generation_kwargs = {
                            "max_new_tokens": max_new_tokens_for_final_summary,
                            "do_sample": False,
                            "temperature": current_model_config.temperature,
                            "pad_token_id": tokenizer_final.eos_token_id,
                            "eos_token_id": tokenizer_final.eos_token_id
                        }
                        with torch.no_grad():
                            outputs = model_final.generate(**inputs, **generation_kwargs)
                        generated_text = tokenizer_final.decode(outputs[0][inputs.input_ids.shape[1]:],
                                                              skip_special_tokens=True)
                    else:  # encoder-decoder
                        inputs = tokenizer_final(example["input_text"], return_tensors="pt",
                                               truncation=True, max_length=current_model_config.max_length).to(model_final.device)
                        with torch.no_grad():
                            outputs = model_final.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens_for_final_summary,
                                min_length=50, # Common for BART summarization
                                length_penalty=2.0, # Common for BART summarization
                                num_beams=4, # Common for BART summarization
                                early_stopping=True, # Common for BART summarization
                                temperature=current_model_config.temperature
                            )
                        generated_text = tokenizer_final.decode(outputs[0], skip_special_tokens=True)

                    generated_summaries_final.append(generated_text.strip())
                    end_time_final = time.time()
                    generation_times_final.append(end_time_final - start_time_final)
                    if (i + 1) % 10 == 0 or i == len(self.test_dataset) - 1:
                        self.logger.info(f"Generated summary {i+1}/{len(self.test_dataset)} for {model_name}. Avg time: {np.mean(generation_times_final):.2f}s")

                except Exception as e_gen:
                    self.logger.error(f"Error generating summary for example {i} with {model_name}: {e_gen}")
                    generated_summaries_final.append("")
                    if "CUDA" in str(e_gen):
                        self.logger.warning("CUDA error during final summary generation. Clearing cache.")
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

            output_data_final = {
                "model_name": model_name,
                "best_trial_params": best_trial_data.get("params"),
                "summaries": generated_summaries_final,
                "references": [ex["target_summary"] for ex in self.test_dataset], # Extract list of strings
                "inputs": [ex["input_text"] for ex in self.test_dataset]      # Extract list of strings
            }

            summaries_dir = Path(f"{self.config.output_dir}/summaries")
            summaries_dir.mkdir(parents=True, exist_ok=True)
            summary_file_path = summaries_dir / f"best_{model_name}_summaries_{self.search_timestamp}.json"
            with open(summary_file_path, 'w') as f:
                json.dump(output_data_final, f, indent=2)
            self.logger.info(f"Saved generated summaries from best {model_name} to {summary_file_path}")

            del model_final, base_model_final, tokenizer_final
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"Cleaned up resources for {model_name} after summary generation.")

        self.logger.info("Finished generating and saving summaries for all best models.")

    def _create_visualizations(self, model_results: Dict[str, Dict], timestamp: str):
        """Create visualizations of hyperparameter search results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set(style="whitegrid")
            
            for model_name, results in model_results.items():
                if 'trials' not in results or not results['trials']:
                    continue
                    
                # Extract data for plotting
                data = []
                for trial in results['trials']:
                    if 'params' not in trial or 'metrics' not in trial:
                        continue
                        
                    row = {
                        'trial': trial['trial_number'],
                        'eval_loss': trial.get('eval_loss', float('inf')),
                        **trial['params'],
                        **{f"metric_{k}": v for k, v in trial.get('metrics', {}).items()}
                    }
                    data.append(row)
                    
                if not data:
                    continue
                    
                df = pd.DataFrame(data)
                
                # Plot hyperparameter relationships
                plt.figure(figsize=(12, 10))
                plt.subplot(2, 2, 1)
                sns.scatterplot(data=df, x='lora_rank', y='eval_loss', hue='batch_size')
                plt.title(f'{model_name} - LoRA Rank vs. Loss')
                
                plt.subplot(2, 2, 2)
                sns.scatterplot(data=df, x='learning_rate', y='eval_loss', hue='batch_size')
                plt.title(f'{model_name} - Learning Rate vs. Loss')
                plt.xscale('log')
                
                plt.subplot(2, 2, 3)
                sns.scatterplot(data=df, x='lora_alpha', y='eval_loss', hue='lora_dropout')
                plt.title(f'{model_name} - LoRA Alpha vs. Loss')
                
                plt.subplot(2, 2, 4)
                sns.scatterplot(data=df, x='warmup_ratio', y='eval_loss', hue='batch_size')
                plt.title(f'{model_name} - Warmup Ratio vs. Loss')
                
                plt.tight_layout()
                plt.savefig(f"{self.config.output_dir}/visualizations/{model_name}_hyperparams_{timestamp}.png")
                
                # Plot metrics
                metrics = [col for col in df.columns if col.startswith('metric_')]
                if metrics:
                    plt.figure(figsize=(14, 8))
                    for i, metric in enumerate(metrics):
                        plt.subplot(2, 3, i+1)
                        sns.boxplot(data=df, x='batch_size', y=metric)
                        plt.title(f'{model_name} - {metric.replace("metric_", "")}')
                        
                    plt.tight_layout()
                    plt.savefig(f"{self.config.output_dir}/visualizations/{model_name}_metrics_{timestamp}.png")
                    
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")
    
    def run_optimization(self):
        """Main method to run hyperparameter optimization"""
        self.logger.info(f"Starting hyperparameter optimization using {self.config.method}")
        
        # Set random seed for reproducibility
        set_seed(self.config.random_seed)
        
        # Login to HuggingFace Hub if token is provided
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(hf_token)
        
        # Get model configurations
        model_configs = self.get_model_configs()
        
        # Run optimization for each model
        all_hyperparam_results = {}
        # self.best_results is populated by random_search via evaluate_trial

        for model_config in model_configs:
            self.logger.info(f"Starting optimization for {model_config.name}")
            
            current_model_hyperparam_search_results = {}
            if self.config.method == "random":
                current_model_hyperparam_search_results = self.random_search(model_config) # This updates self.best_results
                
            all_hyperparam_results[model_config.name] = current_model_hyperparam_search_results
            
        # Save the detailed hyperparameter search results (logs of all trials)
        self.save_hyperparameter_search_details(all_hyperparam_results)

        # Now, generate and save summaries for the best model of each type
        self.generate_and_save_best_model_outputs()
        
        # Prepare the return value for main() function (best params for each model)
        # This can be derived from self.best_results, which is updated by random_search
        best_params_for_main = {}
        for model_name, data in self.best_results.items():
            if data: # if best_results were found for this model
                best_params_for_main[model_name] = {
                    'params': data['params'],
                    'eval_loss': data['eval_loss'],
                    'metrics': data.get('metrics', {}) 
                }
        
        self.logger.info("Hyperparameter optimization and best model summary generation completed.")
        return best_params_for_main

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Medical Text Summarization")
    parser.add_argument("--method", choices=["random"], default="random",
                       help="Optimization method")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--output_dir", default="./results/hyperparameter", help="Output directory")
    parser.add_argument("--dataset_path", default="./data/mimic-iv-bhc.csv", help="Path to the MIMIC-IV-BHC dataset")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum training steps per trial")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of examples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create hyperparameter space
    hp_space = HyperparameterSpace()
    
    # Create search configuration
    search_config = SearchConfig(
        method=args.method,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        max_steps=args.max_steps,
        sample_size=args.sample_size,
        random_seed=args.seed
    )
    
    # Run optimization
    optimizer = MedicalSummarizationOptimizer(hp_space, search_config)
    best_params = optimizer.run_optimization()
    
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*50)
    for model_name, params in best_params.items():
        print(f"\nModel: {model_name}")
        print(f"Best parameters: {params['params']}")
        print(f"Evaluation loss: {params['eval_loss']:.4f}")
        if 'metrics' in params:
            print("Metrics:")
            for metric, value in params['metrics'].items():
                print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()