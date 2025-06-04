#!/bin/bash

# Medical Text Summarization Setup Script for RTX 5090
# =======================================================

set -e # Exit on any error

echo "🏥 Medical Text Summarization Project Setup"
echo "=========================================="

# Check if running on GPU-enabled system
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ No NVIDIA GPU detected. This script is optimized for RTX 5090 GPU."
    echo "Please ensure you have proper NVIDIA drivers installed."
    exit 1
fi

# Create project structure
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p models/{checkpoints,fine-tuned}
mkdir -p results/{baseline,hyperparameter}
mkdir -p logs
mkdir -p notebooks

echo "✅ Directory structure created"

# Install system dependencies
echo "📦 Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y python3-dev build-essential git-lfs 
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install git-lfs
fi

# Setup Python virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support for RTX 5090
echo "🔥 Installing PyTorch with CUDA support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model for text processing
python -m spacy download en_core_web_sm

# Setup NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print('✅ NLTK data downloaded')
"

# Create environment file
cat > .env << 'EOF'
# Environment Configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
TRANSFORMERS_CACHE=./models/cache
HF_HOME=./models/cache

# Data paths
MIMIC_IV_BHC_PATH=./data/

# Model paths
MODEL_CACHE_DIR=./models/cache
CHECKPOINT_DIR=./models/checkpoints
RESULTS_DIR=./results
EOF

# Optimize GPU settings for RTX 5090
echo "⚙️ Optimizing settings for RTX 5090 GPU..."
cat >> .env << 'EOF'
# RTX 5090 GPU Optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TF_GPU_THREAD_MODE=gpu_private
TF_GPU_THREAD_COUNT=1
TF_XLA_FLAGS=--tf_xla_enable_xla_devices
TORCH_EXTENSIONS_DIR=./torch_extensions
EOF

# Add HuggingFace Token placeholder
cat >> .env << 'EOF'
# Add your HuggingFace token here to access Gemma 3 models
# HUGGINGFACE_TOKEN=your_token_here
EOF

# Create shortcut script for baseline evaluation
cat > run_baseline.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python baseline.py "$@"
EOF
chmod +x run_baseline.sh

# Create shortcut script for hyperparameter search
cat > run_hyperparameter_search.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python hyperparameter_search.py "$@"
EOF
chmod +x run_hyperparameter_search.sh

# Make scripts executable
chmod +x baseline.py
chmod +x hyperparameter_search.py

echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "📋 Next steps:"
echo "1. Add your HuggingFace token to .env file to access Gemma 3 models"
echo "2. Place your MIMIC-IV-BHC dataset in the data/ directory"
echo "3. Run baseline evaluation: ./run_baseline.py"
echo "4. Run hyperparameter search: ./run_hyperparameter_search.py"
echo ""
echo "📖 See README.md for detailed documentation"