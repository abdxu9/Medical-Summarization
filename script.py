# Generate a simple visualization showing token distribution in a typical MIMIC dataset
# This is for illustrative purposes since we don't have the actual dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Create simulated token length distributions based on the paper's statistics
np.random.seed(42)

# From the paper: average input tokens = 2,267, average target tokens = 564
input_tokens = np.random.normal(2267, 914, 1000)  # using the standard deviation from the paper
target_tokens = np.random.normal(564, 410, 1000)  # using the standard deviation from the paper

# Ensure positive values
input_tokens = np.maximum(input_tokens, 10)
target_tokens = np.maximum(target_tokens, 10)

# Create a DataFrame
df = pd.DataFrame({
    'input_tokens': input_tokens.astype(int),
    'target_tokens': target_tokens.astype(int),
    'compression_ratio': target_tokens / input_tokens
})

# Plot histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Input token distribution
axes[0].hist(df['input_tokens'], bins=30, color='skyblue', alpha=0.7)
axes[0].set_title('Input Token Distribution (Clinical Notes)')
axes[0].set_xlabel('Number of Tokens')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['input_tokens'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["input_tokens"].mean():.0f}')
axes[0].legend()

# Target token distribution
axes[1].hist(df['target_tokens'], bins=30, color='lightgreen', alpha=0.7)
axes[1].set_title('Target Token Distribution (BHC Summaries)')
axes[1].set_xlabel('Number of Tokens')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['target_tokens'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["target_tokens"].mean():.0f}')
axes[1].legend()

# Compression ratio distribution
axes[2].hist(df['compression_ratio'], bins=30, color='salmon', alpha=0.7)
axes[2].set_title('Compression Ratio Distribution')
axes[2].set_xlabel('Target/Input Ratio')
axes[2].set_ylabel('Frequency')
axes[2].axvline(df['compression_ratio'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["compression_ratio"].mean():.3f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('mimic_iv_bhc_token_distribution.png')
plt.close()

# Create a plot showing model architecture comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Model names and parameters (in billions)
models = ['Gemma 3-12B-IT', 'BART-Large-CNN']
params = [12, 0.4]
colors = ['#4285F4', '#34A853']

# Create bar chart
ax.bar(models, params, color=colors, alpha=0.7)
ax.set_ylabel('Parameters (billions)')
ax.set_title('Model Size Comparison')

# Add text labels
for i, v in enumerate(params):
    ax.text(i, v + 0.1, f"{v}B", ha='center')

# Add annotations for model architecture
ax.annotate('Decoder-only model\nQLoRA fine-tuning', 
            xy=(0, params[0]), xytext=(0, params[0] + 2),
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#D9E7FD", alpha=0.6))

ax.annotate('Encoder-decoder model\nLoRA fine-tuning', 
            xy=(1, params[1]), xytext=(1, params[1] + 2),
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#D6F5D6", alpha=0.6))

plt.savefig('model_comparison.png')
plt.close()

# Create a diagram showing the training pipeline
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# No axes for diagram
ax.axis('off')

# Dataset processing
dataset = Rectangle((0.5, 7), 2, 1.5, fc='#FFF2CC', ec='black')
ax.add_patch(dataset)
ax.text(1.5, 7.75, 'MIMIC-IV-BHC Dataset\n1000 Clinical Notes', ha='center', va='center')

# Split arrow
ax.arrow(2.5, 7.75, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.text(3, 8.1, 'Split 80/10/10', ha='center')

# Train, val, test boxes
train = Rectangle((4, 8.5), 1.5, 0.75, fc='#D4E8D4', ec='black')
val = Rectangle((4, 7.5), 1.5, 0.75, fc='#DAE8FC', ec='black')
test = Rectangle((4, 6.5), 1.5, 0.75, fc='#FFE6CC', ec='black')
ax.add_patch(train)
ax.add_patch(val)
ax.add_patch(test)
ax.text(4.75, 8.875, 'Training Set (800)', ha='center', va='center')
ax.text(4.75, 7.875, 'Validation Set (100)', ha='center', va='center')
ax.text(4.75, 6.875, 'Test Set (100)', ha='center', va='center')

# Baseline evaluation path
baseline_arrow = ax.arrow(5.5, 6.875, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
baseline = Rectangle((7, 6.5), 2, 1, fc='#FFD9D9', ec='black')
ax.add_patch(baseline)
ax.text(8, 7, 'Baseline Evaluation\nROUGE, BERTScore', ha='center', va='center')
ax.text(6, 7.2, 'baseline.py', ha='center', weight='bold')

# Hyperparameter search path
hp_arrow = ax.arrow(5.5, 8.875, 0.5, -1.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
hp_search = Rectangle((6.5, 4), 2, 1.5, fc='#E1D5E7', ec='black')
ax.add_patch(hp_search)
ax.text(7.5, 4.75, 'Hyperparameter Search\nLoRA/QLoRA Random Search', ha='center', va='center')
ax.text(7, 5.7, 'hyperparameter_search.py', ha='center', weight='bold')

# Fine-tuning arrow
ft_arrow = ax.arrow(7.5, 4, 0, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
fine_tuning = Rectangle((6.5, 2), 2, 1, fc='#D5E8D4', ec='black')
ax.add_patch(fine_tuning)
ax.text(7.5, 2.5, 'Fine-tuned Models\nGemma 3-12B-IT & BART', ha='center', va='center')

# Models
gemma_box = Rectangle((2, 3), 2, 1.5, fc='#DAE8FC', ec='black')
bart_box = Rectangle((2, 1), 2, 1.5, fc='#DAE8FC', ec='black')
ax.add_patch(gemma_box)
ax.add_patch(bart_box)
ax.text(3, 3.75, 'Gemma 3-12B-IT\nDecoder-only', ha='center', va='center')
ax.text(3, 1.75, 'BART-Large-CNN\nEncoder-decoder', ha='center', va='center')

# Arrows to fine-tuning
ax.arrow(4, 3.75, 2.5, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(4, 1.75, 2.5, 0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')

plt.savefig('training_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

# Print confirmation
print("Visualizations generated successfully:")
print("1. mimic_iv_bhc_token_distribution.png - Token distributions in the dataset")
print("2. model_comparison.png - Comparison of model architectures")
print("3. training_pipeline.png - Overview of the training pipeline")

# Show paths to the images
import os
current_dir = os.getcwd()
print(f"\nImage files saved to:")
for img in ['mimic_iv_bhc_token_distribution.png', 'model_comparison.png', 'training_pipeline.png']:
    print(f"- {os.path.join(current_dir, img)}")