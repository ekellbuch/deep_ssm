# Deep Sequence Models (deep_ssm)

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
**CRITICAL**: Installation can take 45+ minutes due to large dependencies and potential build compilation. NEVER CANCEL build operations.

#### Method 1: Complete Installation (Recommended)
```bash
# Create conda environment with Python 3.9
conda create -n deep_ssm python=3.9
conda activate deep_ssm

# Install PyTorch with CUDA support - NEVER CANCEL: Takes 10+ minutes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies - NEVER CANCEL: Takes 15+ minutes  
pip install lightning==2.3.3 hydra-core==1.2.0 omegaconf==2.2.3 wandb tqdm einops datasets==2.4.0 transformers==4.42.4 pandas scikit-learn==1.5.1

# Install mamba dependencies (requires CUDA dev tools) - NEVER CANCEL: Takes 20+ minutes
pip install mamba-ssm[causal-conv1d] causal-conv1d triton==2.2.0

# Install package in editable mode
pip install -e src/
```

#### Method 2: Minimal Installation (When full install fails)
```bash
# Use system Python or existing environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install lightning hydra-core omegaconf wandb tqdm einops
pip install -e src/
```

**When to use**: Network timeouts, missing CUDA dev tools, or CI environments with restrictions.

#### Method 4: Emergency Minimal Setup (If most packages fail)
```bash
# Only install absolutely essential packages
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only PyTorch
pip install -e src/
```

**When to use**: Severe network restrictions or build environment issues. Only basic imports will work.

**Note**: If installation fails due to network timeouts or missing CUDA tools, use Method 2. Some advanced models (Mamba) will not work, but basic functionality and S5 models will work.

**Common Installation Failures**:
- `ReadTimeoutError: HTTPSConnectionPool... Read timed out`: Network issues, retry or use Method 2
- `NameError: name 'bare_metal_version' is not defined`: CUDA dev tools missing for causal-conv1d
- `subprocess-exited-with-error`: Missing build dependencies, skip problematic packages

#### Method 3: Development Environment (For Sherlock cluster)
```bash
# Load required modules first
ml python/3.9.0 && ml gcc/10.1.0 && ml cudnn/8.9.0.131 && ml load cuda/12.4.1
./setup_env.sh
```

### Data Setup
```bash
# For BCI experiments, set data environment variable
export DEEP_SSM_DATA=/path/to/data

# Download BCI data (if needed)
gsutil cp gs://cfan/interspeech24/brain2text_competition_data.pkl .

# On Sherlock cluster, data is pre-available:
export DEEP_SSM_DATA=/scratch/groups/swl1
```

### Running Training

#### S5 Model on Sequential CIFAR10
```bash
# Basic sequential CIFAR10 training - NEVER CANCEL: Takes 30+ minutes per epoch
python -m example

# Grayscale version (faster convergence)
python -m example --grayscale

# With wandb logging
python -m example --grayscale --wandb

# MNIST variant
python -m example --dataset mnist --d_model 256 --weight_decay 0.0
```

#### BCI Models with Hydra Configuration
```bash
# Debug run (quick test) - Takes 2-3 minutes
python run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1

# Full GRU training - NEVER CANCEL: Takes 60+ minutes
python run.py --config-name="baseline_gru"

# Mamba model training (requires full installation) - NEVER CANCEL: Takes 90+ minutes  
python run.py --config-name="baseline_mamba"
```

#### Safari Models (Advanced)
```bash
# Safari training - NEVER CANCEL: Takes 45+ minutes
python scripts/train_safari.py
```

## Validation

### Always Run These Tests After Making Changes
```bash
# Test basic imports (should complete in <30 seconds)
python -c "import deep_ssm; print('Package installed correctly')"

# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test configuration loading (only if omegaconf installed)
python -c "
try:
    from omegaconf import OmegaConf
    print('Hydra configs working')
except ImportError:
    print('omegaconf not available - use Method 1 installation')
"

# Check core dependencies availability
python -c "
try:
    import lightning
    print('Lightning available')
except ImportError:
    print('Lightning NOT available - some training scripts will fail')
"

# Run basic mixer tests (only if all dependencies available)
# NOTE: This will fail if einops, absl, or other test dependencies are missing
python tests/mixers/test_mixers.py
```

### Manual Validation Scenarios
**CRITICAL**: Always test one complete training scenario after making code changes:

1. **Quick S5 Test** (5 minutes):
   ```bash
   python -m example --dataset mnist --epochs 1 --batch_size 50
   ```

2. **BCI Debug Test** (3 minutes):
   ```bash
   python run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1
   ```

3. **Full Training Validation** (30+ minutes - run occasionally):
   ```bash
   python -m example --grayscale --epochs 5
   ```

## Time Expectations and Build Commands

### Timing Guide (NEVER CANCEL these operations)
- **Environment setup**: 15-45 minutes depending on method and network speed
- **PyTorch installation**: 5-15 minutes (2GB+ download)
- **Complete dependency installation**: 20-40 minutes total
- **S5 CIFAR training**: 30 minutes per epoch (250 epochs = ~125 hours total)
- **BCI model training**: 60-90 minutes for full run  
- **Debug runs**: 2-5 minutes
- **Tests**: 30 seconds to 5 minutes
- **Package installation**: 2-3 minutes
- **File operations**: <1 second for basic commands

### Build Timeout Recommendations
- Set timeout to 60+ minutes for full installations
- Set timeout to 30+ minutes for PyTorch installation
- Set timeout to 20+ minutes for individual large packages
- Use 2+ minute timeouts for basic operations

### Common Validation Commands
```bash
# Always run before committing - Takes <2 minutes
python -c "import deep_ssm.mixers.s5_fjax.ssm; print('S5 imports work')"
python scripts/run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1

# Check configuration files
find configs/ -name "*.yaml" | head -5
```

## Repository Structure

### Key Entry Points
- `example.py`: S5 model training on CIFAR10/MNIST
- `scripts/run.py`: BCI model training with Hydra configs  
- `scripts/train_safari.py`: Advanced Safari model training

### Important Directories
- `src/deep_ssm/`: Core package code
- `src/safari/`: Safari models and utilities submodule
- `configs/bci/`: BCI model configurations
- `configs/configs_safari/`: Safari model configurations
- `tests/mixers/`: Unit tests for mixer layers

### Configuration Files
- `configs/bci/baseline_gru.yaml`: GRU model config
- `configs/bci/baseline_mamba.yaml`: Mamba model config
- `requirements.txt`: Python dependencies

## Known Issues and Workarounds

### Installation Problems
- **Network timeouts**: Use Method 2 installation if pip hangs (common in CI environments)
- **CUDA compilation fails**: Skip mamba-ssm installation, use S5 models only - error: "nvcc was not found" or "bare_metal_version not defined"
- **Conda activation errors**: Use `eval "$(conda shell.bash hook)"` before `conda activate`
- **PyTorch version conflicts**: Install PyTorch first, then other dependencies to avoid conflicts
- **torchtext version issues**: May need to skip specific versions or install without version constraints

### Runtime Issues  
- **CUDA out of memory**: Reduce batch_size in configs
- **Hydra output directory**: Outputs go to `./outputs/YY-MM-DD/HH-MM-SS/`
- **Missing DEEP_SSM_DATA**: BCI models require this environment variable

### Model-Specific Notes
- **S5 models**: Work with minimal installation
- **Mamba models**: Require full installation with CUDA dev tools
- **Safari models**: Use separate configuration system

## Performance Notes
- **Expected CIFAR10 accuracy**: 88%+ in 250 epochs
- **BCI dataset**: Large neural time series data
- **GPU recommended**: All models benefit significantly from CUDA
- **Memory usage**: 8GB+ GPU memory for larger models

## Development Workflow
1. Always test installation with: `python -c "import deep_ssm"`
2. Run debug mode first: `trainer_cfg.fast_dev_run=1` 
3. Validate with short runs before full training
4. Monitor GPU memory usage during training
5. Check `./outputs/` directory for results and logs