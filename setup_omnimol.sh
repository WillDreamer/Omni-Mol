#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is available"
        return 0
    else
        echo "Conda is not installed. Please install Conda first."
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        if command -v conda &> /dev/null; then
            echo "Conda is available"
            return 0
        else
            return 1
        fi
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_conda || exit 1
    
    # Create and activate conda environment
    # if not exists, create it
    if ! conda env list | grep -q "omnimol"; then
        print_step "Creating conda environment 'omnimol' with Python 3.10..."
        conda create -n omnimol python=3.10 -y
    else
        print_step "Conda environment 'omnimol' already exists"
    fi
    
    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    conda activate omnimol
    
    # Install package in editable mode
    print_step "Installing omnimol package..."

    conda install -c conda-forge mpi4py openmpi -y
    pip install -r requirements.txt
    
    git submodule update --init --recursive

    cd peft
    pip install -e .
    cd ..

    pip install torch_geometric
    
    pip install torch-scatter==2.1.2
    pip install flash-attn==2.5.0

    # === NEW: install NLTK and download WordNet ===
    print_step "Installing NLTK and downloading WordNet corpus..."
    pip install nltk

    python -c "import nltk, sys; nltk.download('wordnet', quiet=True, raise_on_error=True)"
    # === END NEW ===

    sudo apt install -y tmux
    
    # Install PyTorch with CUDA if available
    if check_cuda; then
        print_step "CUDA detected, checking CUDA version..."
        
        if command -v nvcc &> /dev/null; then
            nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            nvcc_major=$(echo $nvcc_version | cut -d. -f1)
            nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
            
            print_step "Found NVCC version: $nvcc_version"
        fi
        
    else
        print_step "Installing PyTorch without CUDA support..."
        
    fi

    print_step "Downloading data..."
    python model_data_download.py

    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "To activate the environment, run: conda activate omnimol"
    
}

# Run main installation
main
