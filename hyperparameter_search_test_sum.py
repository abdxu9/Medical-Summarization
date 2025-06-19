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
from dataclasses import asdict, dataclass, field
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
import optuna # Add Optuna import

# Load environment variables
load_dotenv()
hf_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
if hf_read_token:
    login(token=hf_read_token)

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
    batch_size_choices: List[int] = field(default_factory=lambda: [1])
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
    n_trials: int = 15      # Number of trials to run
    output_dir: str = "./results/hyperparameter"
    dataset_path: str = "./data/mimic-iv-bhc.csv"
    sample_size: Optional[int] = 500  # Using 1000 examples as requested
    eval_steps: int = 50
    max_steps: int = 200    # Limit for each trial
    save_strategy: str = "steps"
    gradient_checkpointing: bool = True
    max_memory_per_gpu: Optional[str] = "30GiB"  # For RTX 5090
    random_seed: int = 42
    # num_train, num_val, num_test will be calculated based on sample_size
    # Ratios for splitting data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is implicitly (1.0 - train_ratio - val_ratio)

    # Fields to be computed
    num_train: int = field(init=False)
    num_val: int = field(init=False)
    num_test: int = field(init=False)

    def __post_init__(self):
        if self.sample_size is None or self.sample_size <= 0:
            # Fallback to a default small configuration or raise error if sample_size is not usable
            # For now, assuming CLI provides a positive sample_size.
            # If sample_size could be None or 0, define behavior e.g.
            # self.num_train, self.num_val, self.num_test = 0,0,0
            # and let downstream code handle empty datasets if that's intended.
            # Or set to some minimal defaults:
            # self.num_train = 1; self.num_val = 0; self.num_test = 0; if self.sample_size == 1 else ...
             raise ValueError("sample_size must be a positive integer for splitting.")


        self.num_train = int(self.sample_size * self.train_ratio)
        self.num_val = int(self.sample_size * self.val_ratio)

        # Ensure num_train is at least 1 if sample_size is positive and train_ratio > 0
        if self.train_ratio > 0 and self.sample_size > 0 and self.num_train == 0:
            self.num_train = 1
        
        # Ensure num_val is at least 1 if val_ratio > 0, and there's enough sample_size
        # after allocating for num_train, and the calculation resulted in 0.
        if self.val_ratio > 0 and (self.sample_size - self.num_train) > 0 and self.num_val == 0:
             self.num_val = 1 # Try to allocate at least one for validation
             # if it makes sum > sample_size, it will be corrected below

        # Remainder goes to test
        self.num_test = self.sample_size - self.num_train - self.num_val

        # Adjust if sum exceeds sample_size due to flooring and minimum allocations
        # (e.g., if num_train=1, num_val=1 for sample_size=1)
        current_sum = self.num_train + self.num_val + self.num_test
        if current_sum > self.sample_size:
            # Reduce from test first, then val, ensuring train is prioritized.
            reduction = current_sum - self.sample_size
            self.num_test -= reduction
            if self.num_test < 0:
                self.num_val += self.num_test # Add negative num_test to num_val (effectively reducing num_val)
                self.num_test = 0
                if self.num_val < 0: # Should not happen if num_train is protected
                    self.num_train += self.num_val
                    self.num_val = 0
        
        # Ensure no split is negative
        self.num_train = max(0, self.num_train)
        self.num_val = max(0, self.num_val)
        self.num_test = max(0, self.num_test)
        
        # Final check: sum of splits should equal sample_size
        # Re-assign num_test to ensure this, prioritizing train and val.
        self.num_test = self.sample_size - self.num_train - self.num_val
        if self.num_test < 0: # This implies num_train + num_val > sample_size
            # This should ideally not happen if num_train and num_val were calculated correctly from ratios.
            # This situation occurs if sample_size is too small for the minimums we enforced.
            # Example: sample_size = 1. num_train = 1. num_val=0. num_test = 0. (Correct)
            self.num_val = self.sample_size - self.num_train 
            if self.num_val < 0 : self.num_val = 0
            self.num_test = 0


        self.logger = logging.getLogger(__name__) # Get logger if not already defined
        self.logger.info(f"Data splits calculated from sample_size {self.sample_size}: "
                         f"Train={self.num_train}, Val={self.num_val}, Test={self.num_test}")
        if self.num_train + self.num_val + self.num_test != self.sample_size:
            self.logger.warning(f"Sum of splits ({self.num_train + self.num_val + self.num_test}) "
                                f"does not equal sample_size ({self.sample_size}). Review split logic.")


class MedicalSummarizationOptimizer:
    """Main class for hyperparameter optimization"""
    def __init__(self, hp_space: HyperparameterSpace, search_config: SearchConfig):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.hp_space = hp_space
        self.config = search_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_logging()
        self.setup_directories()
        
        self.train_dataset, self.eval_dataset, self.test_dataset = self.prepare_dataset()
        
        self.best_results = {}

        self.prompt_template = """You are a doctor in a hospital. You must summarize the patient's medical history, making sure to highlight the key elements so that our peers can quickly understand the situation, background, assessment, and recommendations regarding the patient.

Patient Record:

{input_text}

Summary:"""

        self.logger.info("Optimizer initialized.")

        
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
        Path(f"{self.config.output_dir}/summaries").mkdir(parents=True, exist_ok=True)
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
                max_length=4096,  # Limiting context for efficiency
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                use_quantization=True  # Using QLoRA for Gemma
            ),
            #ModelConfig(
            #    name="bart-large-cnn",
            #    model_type="encoder-decoder",
            #    model_path="facebook/bart-large-cnn",
            #    max_length=1024,
            #    lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            #),
            #ModelConfig(
            #    name="openbiollm-8b",
            #    model_type="decoder",
            #    model_path="aaditya/Llama3-OpenBioLLM-8B",
            #    max_length=8192, # Defaulting to similar as gemma-3
            #    # lora_target_modules will be set by __post_init__ to default decoder modules
            #    use_quantization=True # Defaulting to True as it's a decoder
            #),
            #ModelConfig(
            #    name="long-t5",
            #    model_type="encoder-decoder",
            #    model_path="google/long-t5-tglobal-base",
            #    max_length=4096,
            #    lora_target_modules=["q", "k", "v", "o", "wi", "wo"],
            #    use_quantization=False
            #),
            ModelConfig(
                name="led-base",
                model_type="encoder-decoder",
                model_path="allenai/led-base-16384",
                max_length=4096,
                # lora_target_modules will be set by __post_init__
                # to default encoder-decoder modules (e.g., ["q_proj", "k_proj", ...])
                # which are suitable for LED's BART-like architecture.
                use_quantization=False
            ),
            #ModelConfig(
            #    name="phi-3-medium-4k",
            #    model_type="decoder",
            #    model_path="microsoft/Phi-3-medium-4k-instruct",
            #    max_length=4096,
            #    # lora_target_modules will be set by __post_init__ to default decoder modules
            #    use_quantization=True
            #)
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

    def _objective_bayesian(self, trial: optuna.trial.Trial, model_config: ModelConfig) -> float:
        """Objective function for Optuna Bayesian optimization."""
        trial_params = {
            "lora_rank": trial.suggest_int("lora_rank", self.hp_space.lora_rank_min, self.hp_space.lora_rank_max),
            "lora_alpha": trial.suggest_int("lora_alpha", self.hp_space.lora_alpha_min, self.hp_space.lora_alpha_max),
            "lora_dropout": trial.suggest_float("lora_dropout", self.hp_space.lora_dropout_min, self.hp_space.lora_dropout_max),
            "learning_rate": trial.suggest_float("learning_rate", self.hp_space.learning_rate_min, self.hp_space.learning_rate_max, log=True),
            "batch_size": trial.suggest_categorical("batch_size", self.hp_space.batch_size_choices),
            "warmup_ratio": trial.suggest_float("warmup_ratio", self.hp_space.warmup_ratio_min, self.hp_space.warmup_ratio_max)
        }

        # Ensure lora_alpha is at least lora_rank, if desired, or handle constraints.
        # For now, we sample them independently as per HyperparameterSpace.
        # A common heuristic is lora_alpha = 2 * lora_rank. If you want to enforce this:
        # trial_params["lora_alpha"] = 2 * trial_params["lora_rank"]
        # However, this makes lora_alpha dependent rather than a directly optimized hyperparameter.
        # Sticking to independent sampling based on defined min/max for now.

        self.logger.info(f"Optuna trial {trial.number} for {model_config.name} with sampled params: {trial_params}")
        
        eval_loss, metrics = self.evaluate_trial(model_config, trial_params)

        # Store full metrics and params in Optuna trial's user_attrs for later retrieval
        # This is useful if evaluate_trial doesn't save every trial's full metrics externally.
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params_dict", trial_params) # Store the actual params used by evaluate_trial

        return eval_loss # Optuna minimizes this value
        
    def evaluate_trial(self, model_config: ModelConfig, trial_params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Train and evaluate a model with the given hyperparameters"""
        trial_id = f"{model_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting trial {trial_id} with params: {trial_params}")
        
        model, tokenizer = self.create_model_with_lora(model_config, trial_params)
        
        train_dataset_tokenized = self.tokenize_dataset(self.train_dataset, tokenizer, model_config)
        eval_dataset_tokenized = self.tokenize_dataset(self.eval_dataset, tokenizer, model_config)
        
        if model_config.model_type == "decoder":
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
        else:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
            
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/models/{trial_id}",
            num_train_epochs=3,
            per_device_train_batch_size=trial_params["batch_size"],
            per_device_eval_batch_size=trial_params["batch_size"],
            gradient_accumulation_steps=max(1, 8 // trial_params["batch_size"]),
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
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            tf32=True,
            group_by_length=True,
            optim="paged_adamw_8bit"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=eval_dataset_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        try:
            trainer.train()
            
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss", float('inf'))
            
            metrics = {"eval_loss": eval_loss}
            
            current_eval_loss_for_best_model_check = eval_loss
            if model_config.name not in self.best_results or \
               current_eval_loss_for_best_model_check < self.best_results[model_config.name]["eval_loss"]:
                best_model_path = f"{self.config.output_dir}/models/best_{model_config.name}"
                self.logger.info(f"Saving new best model for {model_config.name} to {best_model_path} (eval_loss: {current_eval_loss_for_best_model_check:.4f})")
                trainer.save_model(best_model_path)
                self.best_results[model_config.name] = {
                    "trial_id": trial_id,
                    "params": trial_params,
                    "eval_loss": current_eval_loss_for_best_model_check, 
                    "metrics": metrics
                }
            
            return eval_loss, metrics
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial_id}: {e}")
            return float('inf'), {"error": str(e)}
        finally:
            # Clean up safely to prevent memory leaks between trials
            self.logger.info("Entering finally block for cleanup in evaluate_trial.")
            
            # Deleting the trainer first, as it holds references to model, tokenizer, and data_collator
            if 'trainer' in locals():
                del trainer
                self.logger.info("Trainer deleted.")
            
            # The data collator can also hold a reference to the model
            if 'data_collator' in locals():
                del data_collator
                self.logger.info("Data collator deleted.")

            if 'model' in locals():
                if self.device == "cuda":
                    model.to('cpu')
                del model
                self.logger.info("Model moved to CPU and deleted.")
            
            if 'tokenizer' in locals():
                del tokenizer
                self.logger.info("Tokenizer deleted.")

            if 'train_dataset_tokenized' in locals():
                del train_dataset_tokenized
                self.logger.info("Tokenized train dataset deleted.")

            if 'eval_dataset_tokenized' in locals():
                del eval_dataset_tokenized
                self.logger.info("Tokenized eval dataset deleted.")

            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Final GPU cache clear in evaluate_trial.")
            
    def random_search(self, model_config: ModelConfig):
        """Perform random search hyperparameter optimization"""
        self.logger.info(f"Starting random search for {model_config.name}")
        
        results = []
        best_trial_so_far = None
        best_value = float('inf')
        
        for trial_idx in range(self.config.n_trials):
            trial_params = self.sample_hyperparameters()
            self.logger.info(f"Trial {trial_idx+1}/{self.config.n_trials} for {model_config.name}")
            
            eval_loss, metrics = self.evaluate_trial(model_config, trial_params)
            
            result = {"trial_number": trial_idx, "params": trial_params, "eval_loss": eval_loss, "metrics": metrics}
            results.append(result)
            
            if eval_loss < best_value:
                best_value = eval_loss
                best_trial_so_far = result
                
        self.logger.info(f"Random search completed for {model_config.name}")
        if best_trial_so_far:
            self.logger.info(f"Best trial for {model_config.name} is {best_trial_so_far['trial_number']} with eval_loss: {best_trial_so_far['eval_loss']:.4f}")

        return {"trials": results, "best_trial": best_trial_so_far}

    def bayesian_search(self, model_config: ModelConfig):
        """Perform Bayesian hyperparameter optimization using Optuna."""
        self.logger.info(f"Starting Bayesian search for {model_config.name} using Optuna")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.config.random_seed))
        
        objective_with_model_config = lambda trial: self._objective_bayesian(trial, model_config)
        
        study.optimize(objective_with_model_config, n_trials=self.config.n_trials)
        
        self.logger.info(f"Bayesian search completed for {model_config.name}")

        processed_trials = []
        for optuna_trial_obj in study.trials:
            trial_data = {
                "trial_number": optuna_trial_obj.number,
                "params": optuna_trial_obj.user_attrs.get("params_dict", optuna_trial_obj.params),
                "eval_loss": optuna_trial_obj.value,
                "metrics": optuna_trial_obj.user_attrs.get("metrics", {})
            }
            processed_trials.append(trial_data)

        best_trial_for_return = None
        if study.best_trial:
            self.logger.info(f"Optuna Best Trial Found: Number {study.best_trial.number} with eval_loss {study.best_trial.value:.4f}")
            best_trial_for_return = {
                "trial_number": study.best_trial.number,
                "params": study.best_trial.params,
                "eval_loss": study.best_trial.value,
                "metrics": study.best_trial.user_attrs.get("metrics", {})
            }

        return {"trials": processed_trials, "best_trial": best_trial_for_return}
    
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
        
        
        
        # Get model configurations
        model_configs = self.get_model_configs()
        
        # Run optimization for each model
        all_hyperparam_results = {}
        # self.best_results is populated by random_search via evaluate_trial

        for model_config in model_configs:
            self.logger.info(f"Starting optimization for {model_config.name}")
            
            current_model_hyperparam_search_results = {}
            if self.config.method == "random":
                current_model_hyperparam_search_results = self.random_search(model_config)
            elif self.config.method == "bayesian":
                current_model_hyperparam_search_results = self.bayesian_search(model_config)
            else:
                self.logger.error(f"Unsupported optimization method: {self.config.method}")
                continue
                
            all_hyperparam_results[model_config.name] = current_model_hyperparam_search_results
            
        # Save the detailed hyperparameter search results (logs of all trials)
        self.save_hyperparameter_search_details(all_hyperparam_results)

        # Score calculation and summary generation has been removed.
        
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
        return best_params_for_main

    def generate_summaries(self, num_summaries: int = 2):
        """Generate summaries for the best performing model of each architecture."""
        self.logger.info(f"Starting summary generation for {num_summaries} samples.")

        if not self.test_dataset or len(self.test_dataset) == 0:
            self.logger.warning("Test dataset is empty. Skipping summary generation.")
            return

        if len(self.test_dataset) < num_summaries:
            self.logger.warning(f"Requested {num_summaries} summaries, but test set only has {len(self.test_dataset)} samples. Generating {len(self.test_dataset)} instead.")
            num_summaries = len(self.test_dataset)

        test_samples = self.test_dataset.shuffle(seed=self.config.random_seed).select(range(num_summaries))
        model_configs = self.get_model_configs()

        for model_config in model_configs:
            best_model_path = f"{self.config.output_dir}/models/best_{model_config.name}"
            if not os.path.isdir(best_model_path):
                self.logger.warning(f"No best model found for {model_config.name} at {best_model_path}. Skipping summary generation for this model.")
                continue

            self.logger.info(f"Generating summaries for {model_config.name} using model from {best_model_path}")

            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if model_config.use_quantization and model_config.model_type == "decoder":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16

            if model_config.model_type == "encoder-decoder":
                base_model = AutoModelForSeq2SeqLM.from_pretrained(model_config.model_path, **model_kwargs)
            else:
                if "gemma-3" in model_config.model_path.lower():
                    base_model = Gemma3ForConditionalGeneration.from_pretrained(model_config.model_path, **model_kwargs)
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(model_config.model_path, **model_kwargs)
            
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = PeftModel.from_pretrained(base_model, best_model_path)
            model.eval()

            generated_summaries = []
            from tqdm import tqdm
            for sample in tqdm(test_samples, desc=f"Generating summaries for {model_config.name}"):
                input_text = sample['input_text']

                if model_config.model_type == "decoder":
                    prompt = self.prompt_template.format(input_text=input_text)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model_config.max_length).to(self.device)
                else:
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=model_config.max_length).to(self.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=model_config.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0]
                if model_config.model_type == "decoder":
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = generated_tokens[input_length:]

                generated_summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_summaries.append(generated_summary.strip())
            
            best_hyperparams_for_model = self.best_results.get(model_config.name, {})

            output_data = {
                "model_config": asdict(model_config),
                "best_hyperparameters": best_hyperparams_for_model,
                "summaries": generated_summaries,
                "references": test_samples["target_summary"],
                "inputs": test_samples["input_text"]
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"{self.config.output_dir}/summaries/{model_config.name}_summaries_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            self.logger.info(f"Saved {len(generated_summaries)} summaries for {model_config.name} to {summary_file}")
            
            del model, base_model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.logger.info("Summary generation finished.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Medical Text Summarization")
    parser.add_argument("--method", choices=["random", "bayesian"], default="random",
                       help="Optimization method (random or bayesian)")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--output_dir", default="./results/hyperparameter", help="Output directory")
    parser.add_argument("--dataset_path", default="./data/mimic-iv-bhc.csv", help="Path to the MIMIC-IV-BHC dataset")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum training steps per trial")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of examples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_summaries", type=int, default=30, help="Number of summaries to generate from the test set for each best model. 0 to disable.")
    
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

    if args.num_summaries > 0:
        optimizer.generate_summaries(num_summaries=args.num_summaries)
    
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