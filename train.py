#!/usr/bin/env python3

"""
Final Model Training Script
===========================

This script trains a model on the full dataset using a specified set of
hyperparameters, typically the best ones found during a hyperparameter search.

It supports training for:
- gemma-3-12b-it
- led-base
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ML Libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
    Gemma3ForConditionalGeneration, set_seed, EarlyStoppingCallback
)
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

# Load environment variables
load_dotenv()
hf_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
if hf_read_token:
    login(token=hf_read_token)

# --- Model and Tokenizer Setup ---

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Returns the static configuration for a given model name."""
    configs = {
        "gemma-3-12b-it": {
            "model_type": "decoder",
            "model_path": "google/gemma-3-12b-it",
            "max_length": 4096,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "use_quantization": True,
        },
        "led-base": {
            "model_type": "encoder-decoder",
            "model_path": "allenai/led-base-16384",
            "max_length": 4096,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "use_quantization": False,
        }
    }
    if model_name not in configs:
        raise ValueError(f"Model {model_name} is not supported. Choose from {list(configs.keys())}")
    return configs[model_name]


def create_model_with_lora(model_name: str, model_config: Dict[str, Any], hyperparams: argparse.Namespace):
    """Creates a model with a LoRA adapter based on the provided hyperparameters."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {model_name} with LoRA (rank={hyperparams.lora_rank})")

    bnb_config = None
    if model_config["use_quantization"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        logger.info("Using 4-bit quantization")

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config["model_type"] == "encoder-decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_config["model_path"], **model_kwargs)
        task_type = TaskType.SEQ_2_SEQ_LM
    elif model_config["model_type"] == "decoder":
        model_class = Gemma3ForConditionalGeneration if "gemma-3" in model_config["model_path"] else AutoModelForCausalLM
        model = model_class.from_pretrained(model_config["model_path"], **model_kwargs)
        task_type = TaskType.CAUSAL_LM
    else:
        raise ValueError(f"Unsupported model_type: {model_config['model_type']}")

    lora_config = LoraConfig(
        task_type=task_type,
        r=hyperparams.lora_rank,
        lora_alpha=hyperparams.lora_alpha,
        lora_dropout=hyperparams.lora_dropout,
        target_modules=model_config["lora_target_modules"]
    )
    model = get_peft_model(model, lora_config)
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    return model, tokenizer

# --- Data Processing ---

PROMPT_TEMPLATE = """You are a doctor in a hospital. You must summarize the patient's medical history, making sure to highlight the key elements so that our peers can quickly understand the situation, background, assessment, and recommendations regarding the patient.

Patient Record:

{input_text}

Summary:"""

def tokenize_dataset(dataset: Dataset, tokenizer, model_config: Dict[str, Any]) -> Dataset:
    """Tokenizes the dataset according to the model type."""
    max_len = model_config["max_length"]

    def tokenize_function(examples):
        if model_config["model_type"] == "decoder":
            formatted_inputs = [PROMPT_TEMPLATE.format(input_text=text) for text in examples["input_text"]]
            full_texts = [f"{prompt}{summary}{tokenizer.eos_token}" for prompt, summary in zip(formatted_inputs, examples["target_summary"])]
            
            model_inputs = tokenizer(full_texts, max_length=max_len, padding="max_length", truncation=True)
            
            # The 'labels' are the same as the 'input_ids' for language modeling.
            # The loss will be calculated only on the summary part by the trainer.
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        else: # encoder-decoder
            model_inputs = tokenizer(examples["input_text"], max_length=max_len, padding="max_length", truncation=True)
            labels = tokenizer(text_target=examples["target_summary"], max_length=512, padding="max_length", truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset


def generate_summaries(args: argparse.Namespace, model_config: Dict[str, Any], final_model_path: str, test_dataset: Dataset):
    from tqdm import tqdm
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_to_generate = min(args.num_summaries, len(test_dataset))
    if num_to_generate == 0:
        logger.info("Test set is empty, skipping summary generation.")
        return

    logger.info(f"Generating {num_to_generate} summaries from the test set...")
    test_samples = test_dataset.shuffle(seed=args.seed).select(range(num_to_generate))

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if model_config["use_quantization"]:
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

    if model_config["model_type"] == "encoder-decoder":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_config["model_path"], **model_kwargs)
    else:
        model_class = Gemma3ForConditionalGeneration if "gemma-3" in model_config["model_path"] else AutoModelForCausalLM
        base_model = model_class.from_pretrained(model_config["model_path"], **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(base_model, final_model_path)
    model.eval()

    generated_summaries = []
    for sample in tqdm(test_samples, desc=f"Generating summaries for {args.model_name}"):
        input_text = sample['input_text']

        if model_config["model_type"] == "decoder":
            prompt = PROMPT_TEMPLATE.format(input_text=input_text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model_config["max_length"]).to(device)
        else:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=model_config["max_length"]).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0]
        if model_config["model_type"] == "decoder":
            input_length = inputs.input_ids.shape[1]
            generated_tokens = generated_tokens[input_length:]

        generated_summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_summaries.append(generated_summary.strip())

    output_path = Path(args.output_dir) / args.model_name
    output_data = {
        "model_name": args.model_name,
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summaries": generated_summaries,
        "references": test_samples["target_summary"],
        "inputs": test_samples["input_text"]
    }

    summary_file = output_path / f"generated_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved {len(generated_summaries)} summaries to {summary_file}")
    
    del model, base_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Main Training Execution ---

def train(args: argparse.Namespace):
    """Main function to run the training process."""
    # Setup
    set_seed(args.seed)
    output_path = Path(args.output_dir) / args.model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training for {args.model_name} with parameters: {vars(args)}")

    # Load and split dataset
    logger.info("Loading dataset...")
    full_df = pd.read_csv(args.dataset_path).rename(columns={'input': 'input_text', 'target': 'target_summary'})
    full_df = full_df[['input_text', 'target_summary']].dropna().reset_index(drop=True)

    if args.sample_size:
        logger.info(f"Using a sample of {args.sample_size} examples for this run.")
        df = full_df.sample(n=args.sample_size, random_state=args.seed)
    else:
        df = full_df

    # Shuffle and split the dataset
    df_shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if len(df_shuffled) < 3:
        # Not enough data for a 3-way split, prioritize train and val
        logger.warning(f"Dataset has only {len(df_shuffled)} samples. Creating train and validation sets only.")
        if len(df_shuffled) == 2:
            train_df = df_shuffled.iloc[:1]
            val_df = df_shuffled.iloc[1:]
            test_df = df_shuffled.iloc[0:0] # empty
        else: # 1 or 0 samples
            train_df = df_shuffled
            val_df = df_shuffled.iloc[0:0] # empty
            test_df = df_shuffled.iloc[0:0] # empty
    else:
        # 80/10/10 split, ensuring val and test have at least one sample
        test_size = max(1, int(0.1 * len(df_shuffled)))
        val_size = max(1, int(0.1 * len(df_shuffled)))
        train_size = len(df_shuffled) - val_size - test_size
        
        train_df = df_shuffled.iloc[:train_size]
        val_df = df_shuffled.iloc[train_size:train_size + val_size]
        test_df = df_shuffled.iloc[train_size + val_size:]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test examples.")

    # Create model and tokenizer
    model_config = get_model_config(args.model_name)
    model, tokenizer = create_model_with_lora(args.model_name, model_config, args)

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, model_config)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, model_config)

    # Configure Trainer
    if model_config["model_type"] == "decoder":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    # Train and save
    logger.info("Starting training...")
    trainer.train()

    # --- Log and Save Training History ---
    logger.info("Training complete. Saving training history...")
    
    # Convert the log history to a pandas DataFrame for easier analysis
    history_df = pd.DataFrame(trainer.state.log_history)
    
    # Save the full history to a CSV file
    history_path = output_path / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    
    # Also print the validation loss history to the console
    eval_history_df = history_df[history_df['eval_loss'].notna()].copy()
    logger.info("Validation Loss per Epoch:")
    print(eval_history_df[['epoch', 'eval_loss']].to_string(index=False))
    # --- End of History Logging ---
    
    final_model_path = str(output_path / "final_adapter")
    logger.info(f"Training complete. Saving final model adapter to {final_model_path}")
    trainer.save_model(final_model_path)
    
    # Save the hyperparameters used for this training run
    with open(output_path / "training_hyperparameters.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.num_summaries > 0 and test_dataset:
        generate_summaries(args, model_config, final_model_path, test_dataset)

    logger.info("Script finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train a summarization model with specific hyperparameters.")
    
    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, required=True, choices=["gemma-3-12b-it", "led-base"], help="The model to train.")
    parser.add_argument("--dataset_path", type=str, default="./data/mimic-iv-bhc.csv", help="Path to the training data.")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of examples to use for training/validation. Uses all data if not set.")
    parser.add_argument("--output_dir", type=str, default="./results/final_models", help="Directory to save the final model adapter.")

    # Hyperparameter Arguments
    parser.add_argument("--lora_rank", type=int, required=True, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, required=True, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, required=True, help="LoRA dropout.")
    parser.add_argument("--learning_rate", type=float, required=True, help="The learning rate for the AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, required=True, help="Per-device training batch size.")
    parser.add_argument("--warmup_ratio", type=float, required=True, help="Warmup ratio for the learning rate scheduler.")
    
    # Training Control Arguments
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Maximum number of training epochs to run.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If set, overrides num_train_epochs. Ideal for testing.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop training if eval_loss does not improve for this many epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_summaries", type=int, default=0, help="Number of summaries to generate from the test set after training. 0 to disable.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()