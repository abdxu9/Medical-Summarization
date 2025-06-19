# Medical Text Summarization using Large Language Models

This repository offers a complete, research‑grade framework for the development and systematic evaluation of large language models (LLMs) that generate concise "Brief Hospital Course" (BHC) summaries from discharge notes contained in the **MIMIC‑IV‑BHC** corpus.  The overarching objective is to accelerate clinical comprehension by producing Situation–Background–Assessment–Recommendation (SBAR) narratives that distil each patient’s hospital trajectory into a form that is immediately intelligible to busy healthcare professionals.

## Project Overview

The workflow embodied in this repository proceeds through five sequential phases—baseline benchmarking, hyper‑parameter optimisation, full‑scale fine‑tuning, independent evaluation, and dissemination.  Each phase is encapsulated in a dedicated script whose behaviour can be controlled from the command line, thereby permitting reproducible experimentation.  During optimisation we employ Low‑Rank Adaptation (LoRA) and its quantised variant (QLoRA) in order to minimise computational cost without compromising linguistic fidelity.

## Models Supported

The current implementation accommodates six state‑of‑the‑art language‑generation architectures.  Three are decoder‑only transformers—**google/gemma‑3‑12b‑it**, **microsoft/Phi‑3‑medium‑4k‑instruct**, and **aaditya/Llama3‑OpenBioLLM‑8B**—while the remaining three are encoder–decoder models: **facebook/bart‑large‑cnn**, **allenai/led‑base‑16384**, and **google/long‑t5‑tglobal‑base**.  The modular training loop can be readily extended to additional checkpoints by supplying appropriately formatted configuration files.

## Dataset

All experiments rely on the *MIMIC‑IV‑BHC* dataset, which consists of 270 033 de‑identified discharge notes (mean length ≈ 2 267 tokens) paired with reference BHC summaries (mean length ≈ 564 tokens).  By default, each script operates on a stratified subsample to accelerate prototyping; a single flag change activates the full dataset for definitive training.

## Setup and Installation

### Hardware requirements

A single **NVIDIA GPU** offering at least 24 GB of dedicated VRAM (for example an RTX 3090, 4090, or 5090) is strongly recommended.  Systems should also provide ≥ 32 GB of system RAM and no less than 50 GB of free local storage.

### Environment initialisation

Clone the repository and enter the working directory:

```bash
git clone https://github.com/yourusername/medical-text-summarization.git
cd medical-text-summarization
```

Execute the setup script, which creates an isolated Python virtual environment, installs all dependencies, and prepares the required folder hierarchy:

```bash
chmod +x setup.sh
./setup.sh
```

Activate the environment:

```bash
source venv/bin/activate
```

Add your Hugging Face credentials to the newly generated `.env` file:

```dotenv
HUGGINGFACE_READ_TOKEN="hf_..."
HUGGINGFACE_WRITE_TOKEN="hf_..."
```

Finally, download the dataset and place it at `data/mimic-iv-bhc.csv`.

## End‑to‑End Workflow

The canonical experimental sequence is outlined below.  Each step is fully automated yet customisable via command‑line arguments.

### Step 1  Baseline evaluation

Generate naïve summaries and compute ROUGE and BERTScore values:

```bash
python baseline.py --sample_size 100 --batch_size 4
```

The `--sample_size` argument (default 1000) controls the number of discharge notes evaluated, whereas `--batch_size` (default 4) sets the generation batch size.  Results are stored under `./results/baseline/`.

### Step 2  Hyper‑parameter optimisation

Invoke Optuna‑based Bayesian search (or fallback random search) to identify optimal LoRA/QLoRA settings:

```bash
python hyperparameter_search_test_sum.py \
       --model_name "gemma-3-12b-it" \
       --n_trials 15 \
       --method "bayesian" \
       --sample_size 500 \
       --max_steps 200
```

Comprehensive logs and the highest‑scoring adapter checkpoint are written to `./results/hyperparameter/`.

### Step 3  Final model training

Train the chosen architecture on the full dataset (or an expanded subsample) using the best hyper‑parameters:

```bash
python train.py \
       --model_name "gemma-3-12b-it" \
       --dataset_path "./data/mimic-iv-bhc.csv" \
       --lora_rank 16 \
       --lora_alpha 32 \
       --lora_dropout 0.1 \
       --learning_rate 5e-5 \
       --batch_size 1 \
       --warmup_ratio 0.05 \
       --num_train_epochs 3 \
       --num_summaries 100
```

The resulting adapter is deposited in `./results/final_models/<model_name>/`, accompanied by generated summaries for subsequent evaluation.

### Step 4  In‑depth evaluation

Compute BLEU, METEOR, and domain‑specific BERTScore values for a JSON file of generated summaries:

```bash
python evaluate_bertscore.py \
       "./results/final_models/gemma-3-12b-it/generated_summaries_....json" \
       --model_type "Simonlee711/Clinical-Longformer" \
       --output_dir "./results/evaluation"
```

Aggregated and per‑instance statistics are written to the designated directory.

### Step 5  Visualise and compare results

Merge the CSV files created in Step 4 into a unified suite of comparative plots:

```bash
python plot_results.py \
       --results_dir "./results/evaluation" \
       --output_dir "./results/plots"
```

### Step 6  Upload to the Hugging Face Hub

Disseminate the lightweight adapter so that other investigators can reproduce or extend your work:

```bash
python upload.py \
       --model_name "gemma-3-12b-it" \
       --local_adapter_path "./results/final_models/gemma-3-12b-it/final_adapter/" \
       --hub_repo_id "YourUsername/gemma-3-12b-it-mimic-bhc-summarization"
```

Only the adapter weights and tokenizer configuration are transferred, keeping storage overhead minimal.

## Project Structure

```
medical-text-summarization/
├── data/                    # Dataset directory (e.g., mimic-iv-bhc.csv)
├── logs/                    # Log files for baseline and hyperparameter search
├── results/                 # Evaluation results, plots, and models
│   ├── baseline/            # Baseline evaluation metrics and summaries
│   ├── hyperparameter/      # Hyperparameter search results and best models
│   ├── final_models/        # Final trained model adapters and summaries
│   ├── evaluation/          # Detailed evaluation scores (ROUGE, BERTScore, etc.)
│   └── plots/               # Final comparison plots
├── .env                     # Environment variables (incl. HF tokens)
├── baseline.py              # Baseline evaluation script
├── hyperparameter_search_test_sum.py
├── train.py                 # Final model training script
├── evaluate_bertscore.py    # In‑depth evaluation script
├── plot_results.py          # Visualisation script
├── upload.py                # Hub upload script
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script
└── README.md                # Project documentation
```

## Evaluation Metrics

Performance is appraised using four complementary criteria.  ROUGE‑1, ROUGE‑2, and ROUGE‑L quantify lexical overlap at the unigram, bigram, and longest‑common‑subsequence levels, respectively.  BERTScore gauges semantic congruence by exploiting contextual embeddings derived from a domain‑specific transformer.  BLEU, a precision‑oriented machine‑translation measure, assesses n‑gram concurrence, whereas METEOR incorporates stemming, synonym matching, and word‑order penalties to yield a more holistic reflection of quality.

## Dependencies

All software prerequisites are declared in `requirements.txt`.  Principal libraries include **PyTorch**, **Transformers**, **PEFT**, **TRL**, and **BitsAndBytes** for model training; **Evaluate**, **ROUGE**, **BERTScore**, and **NLTK** for metric computation; **Optuna** for Bayesian optimisation; and **Pandas**, **Matplotlib**, and **Seaborn** for statistical analysis and visualisation.

## Acknowledgements

The work rests upon data derived from the *Medical Information Mart for Intensive Care IV* (MIMIC‑IV) database, curated by the MIT Laboratory for Computational Physiology.  All models trained with these data must preserve patient anonymity in accordance with the MIMIC‑IV data‑use agreement.

## License

This project is distributed under the terms of the MIT License.  Please consult the `LICENSE` file for the full text.

