# Medical Text Summarization with Gemma 3-12B-IT and BART-Large-CNN

This repository contains the code and resources for evaluating and fine-tuning large language models (Gemma 3-12B-IT and BART-Large-CNN) for medical history summarization using the MIMIC-IV-BHC dataset.

## Project Overview

The goal of this project is to establish baseline performance and optimize hyperparameters for two language models on a medical summarization task. The models will summarize patient discharge notes into concise hospital course summaries, which can help healthcare professionals quickly understand a patient's situation, background, assessment, and recommendations.

### Models

- **Gemma 3-12B-IT**: A decoder-only model with 12 billion parameters, fine-tuned with QLoRA
- **BART-Large-CNN**: An encoder-decoder model fine-tuned on CNN/DailyMail, adapted with LoRA

### Dataset

We use the MIMIC-IV-BHC (Brief Hospital Course) dataset, which contains:
- 270,033 clinical notes with an average token length of 2,267
- Each note paired with a corresponding BHC summary (average token length of 564)
- The dataset will be split into training (80%), validation (10%), and test (10%) sets
- We'll start with a sample of 1,000 examples for initial experiments

## Setup and Installation

### Hardware Requirements

- GPU: RTX 5090 with 32GB VRAM (or equivalent)
- RAM: 32GB+ recommended
- Storage: 50GB+ free space

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-text-summarization.git
cd medical-text-summarization
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Add your HuggingFace token to the `.env` file to access Gemma 3 models:
```
HUGGINGFACE_TOKEN="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"
```

4. Download and place the MIMIC-IV-BHC dataset in the `data/` directory:
```
data/mimic-iv-bhc.csv
```

## Usage

### Baseline Evaluation

Run the baseline evaluation to establish performance metrics for both models:

```bash
python baseline.py --sample_size 100
```

Optional arguments:
- `--output_dir`: Output directory for results (default: `./results/baseline`)
- `--dataset_path`: Path to the dataset (default: `./data/mimic-iv-bhc.csv`)
- `--sample_size`: Number of examples to evaluate (default: 100)
- `--batch_size`: Batch size for evaluation (default: 4)
- `--max_tokens`: Maximum new tokens to generate (default: 512)
- `--seed`: Random seed (default: 42)
- `--bert_score_model`: Model type for BERTScore evaluation (default: `allenai/longformer-base-4096`)

### Hyperparameter Optimization

Run hyperparameter search to find optimal configurations for fine-tuning:

```bash
python hyperparameter_search.py --n_trials 10
```

Optional arguments:
- `--method`: Optimization method (default: `random`)
- `--n_trials`: Number of trials (default: 10)
- `--output_dir`: Output directory (default: `./results/hyperparameter`)
- `--dataset_path`: Path to the dataset (default: `./data/mimic-iv-bhc.csv`)
- `--max_steps`: Maximum training steps per trial (default: 300)
- `--sample_size`: Number of examples to use (default: 1000)
- `--seed`: Random seed (default: 42)

## Project Structure

```
medical-text-summarization/
├── data/                # Dataset directory
├── models/              # Model checkpoints and fine-tuned models
├── results/             # Evaluation results and visualizations
│   ├── baseline/        # Baseline evaluation results
│   └── hyperparameter/  # Hyperparameter search results
├── logs/                # Log files
├── baseline.py          # Baseline evaluation script
├── hyperparameter_search.py  # Hyperparameter search script
├── requirements.txt     # Python dependencies
├── setup.sh             # Environment setup script
└── README.md            # Project documentation
```

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries
- **ROUGE-L**: Longest common subsequence between generated and reference summaries
- **BERTScore**: Semantic similarity using contextual embeddings

## QLoRA and LoRA Configuration

We use Quantized Low-Rank Adaptation (QLoRA) for memory-efficient fine-tuning of Gemma 3-12B-IT and LoRA for BART-Large-CNN. The key hyperparameters optimized include:

- **LoRA rank (r)**: Number of low-rank factors (4-64)
- **LoRA alpha**: Scaling factor for weight updates (typically rank * 2)
- **LoRA dropout**: Dropout rate to prevent overfitting (0.05-0.3)
- **Learning rate**: Rate of parameter updates (1e-5 to 5e-4)
- **Batch size**: Number of samples processed per step (1, 2, or 4)
- **Warmup ratio**: Portion of training used for learning rate warmup (0.03-0.1)

## Results

After running the baseline evaluation and hyperparameter search, results will be saved in the `results/` directory, including:

- CSV files with summary metrics
- JSON files with detailed results
- Visualizations comparing model performance
- Example summaries for qualitative analysis

## Dependencies

Key dependencies include:

- PyTorch 2.5.0+
- Transformers 4.50.0+
- PEFT 0.13.0+
- BitsAndBytes 0.44.0+
- TRL 0.11.0+
- Evaluate 0.4.1+
- ROUGE and BERTScore for evaluation

See `requirements.txt` for the complete list.

## Acknowledgments

This project uses the MIMIC-IV-BHC dataset, derived from the MIMIC-IV clinical database maintained by the MIT Laboratory for Computational Physiology.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{mimic-iv-bhc,
  title={A Dataset and Benchmark for Hospital Course Summarization with Adapted Large Language Models},
  author={Aali, A. and Van Veen, D. and Arefeen, Y. and others},
  journal={arXiv preprint arXiv:2403.05720},
  year={2024}
}
```