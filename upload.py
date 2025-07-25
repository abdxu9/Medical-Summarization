#!/usr/bin/env python3

"""
Model Upload Script
===================

This script loads a trained PEFT adapter from a local directory,
attaches it to its base model, and pushes only the adapter weights
to the Hugging Face Hub.
"""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
    BitsAndBytesConfig, Gemma3ForConditionalGeneration
)

# --- Prerequisite: Log in to Hugging Face Hub ---
# This script requires a Hugging Face User Access Token with 'write' permissions.
# Ensure HUGGINGFACE_WRITE_TOKEN is set in your .env file.
print("--- Hugging Face Login ---")
load_dotenv()
hf_write_token = os.getenv("HUGGINGFACE_WRITE_TOKEN")
if not hf_write_token:
    raise ValueError(
        "Hugging Face write token not found. Please set HUGGINGFACE_WRITE_TOKEN in your .env file."
    )
login(token=hf_write_token, add_to_git_credential=True)
print("Successfully logged in to Hugging Face Hub with write permissions.")


def get_model_config(model_name: str) -> dict:
    """Returns the static configuration for a given model name."""
    configs = {
        "medgemma": {
            "model_type": "decoder",
            "model_path": "google/medgemma-27b-text-it",
            "use_quantization": True,
        },
        "gemma-3-12b-it": {
            "model_type": "decoder",
            "model_path": "google/gemma-3-12b-it",
            "use_quantization": True,
        },
        "led-base": {
            "model_type": "encoder-decoder",
            "model_path": "allenai/led-base-16384",
            "use_quantization": False,
        },
    }
    if model_name not in configs:
        raise ValueError(
            f"Model {model_name} is not supported. Choose from {list(configs.keys())}"
        )
    return configs[model_name]


def upload_model(args: argparse.Namespace):
    """Loads a base model and adapter, then pushes the adapter to the Hub."""
    print(f"\n--- Starting Upload for Model: {args.model_name} ---")

    # 1. Get model configuration and check adapter path
    model_config = get_model_config(args.model_name)
    local_adapter_path = Path(args.local_adapter_path)

    if not local_adapter_path.exists() or not (local_adapter_path / 'adapter_config.json').exists():
        raise FileNotFoundError(
            f"Adapter path does not exist or is not a valid adapter directory: {local_adapter_path}"
        )
    print(f"Found local adapter at: {local_adapter_path}")

    # 2. Configure model loading arguments (reusing logic from your train.py)
    bnb_config = None
    if model_config["use_quantization"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("INFO: Using 4-bit quantization for base model loading.")

    model_kwargs = {"trust_remote_code": True}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    # 3. Load the base model
    print(f"Loading base model '{model_config['model_path']}'...")
    if model_config["model_type"] == "encoder-decoder":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_config["model_path"], **model_kwargs
        )
    else: # Decoder
        model_class = (
            Gemma3ForConditionalGeneration
            if "gemma-3" in model_config["model_path"]
            else AutoModelForCausalLM
        )
        base_model = model_class.from_pretrained(
            model_config["model_path"], **model_kwargs
        )

    # 4. Load the PEFT model by applying your local adapter to the base model
    print("Applying LoRA adapter to the base model...")
    model = PeftModel.from_pretrained(base_model, str(local_adapter_path))
    print("Adapter applied successfully.")

    # 5. Push the ADAPTER to the Hub. This only uploads the small adapter files.
    print(f"Pushing adapter layers to Hub repository: '{args.hub_repo_id}'")
    model.push_to_hub(args.hub_repo_id)
    print("Adapter push complete.")

    # 6. Push the TOKENIZER to the same Hub repository so it's bundled together.
    print(f"Pushing tokenizer to Hub repository: '{args.hub_repo_id}'")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])
    tokenizer.push_to_hub(args.hub_repo_id)
    print("Tokenizer push complete.")

    print(f"\n✅ Successfully uploaded {args.model_name} to {args.hub_repo_id}")
    print(f"You can view your model at: https://huggingface.co/{args.hub_repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained PEFT adapter to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["medgemma", "gemma-3-12b-it", "led-base"],
        help="The name of the model architecture you are uploading.",
    )
    parser.add_argument(
        "--local_adapter_path",
        type=str,
        required=True,
        help="Path to the folder containing the trained adapter (e.g., './results/final_models/gemma-3-12b-it/final_adapter').",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        required=True,
        help="Name for the repository on the Hub (e.g., 'YourUsername/gemma-3-12b-it-sbar-summary').",
    )

    args = parser.parse_args()
    upload_model(args)


if __name__ == "__main__":
    main()