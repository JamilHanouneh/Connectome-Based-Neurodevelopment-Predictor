#!/usr/bin/env python3
"""
Setup script for Neurodevelopmental Outcome Predictor
Creates directory structure and verifies dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Current Python version: {version_str}")
    print_info(f"Platform: {platform.system()} {platform.machine()}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version {version_str} meets requirements (≥3.8)")
        return True
    else:
        print_error(f"Python version {version_str} is too old")
        print_error("Please install Python 3.8 or higher")
        return False


def create_directory_structure() -> bool:
    """Create project directory structure"""
    print_header("Creating Directory Structure")
    
    directories = [
        # Data directories
        "data/raw",
        "data/processed",
        "data/synthetic",
        "data/clinical",
        
        # Source code directories
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        
        # Output directories
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/predictions",
        "outputs/figures",
        "outputs/reports",
        
        # Notebook directory
        "notebooks",
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory}/")
        
        # Create __init__.py files
        init_files = [
            "src/__init__.py",
            "src/data/__init__.py",
            "src/models/__init__.py",
            "src/training/__init__.py",
            "src/evaluation/__init__.py",
            "src/utils/__init__.py",
        ]
        
        for init_file in init_files:
            Path(init_file).touch(exist_ok=True)
        
        print_success("All __init__.py files created")
        return True
        
    except Exception as e:
        print_error(f"Failed to create directories: {str(e)}")
        return False


def install_dependencies() -> bool:
    """Install required packages from requirements.txt"""
    print_header("Installing Dependencies")
    
    if not Path("requirements.txt").exists():
        print_error("requirements.txt not found!")
        return False
    
    print_info("This may take several minutes...")
    print_info("Installing packages from requirements.txt")
    
    try:
        # Upgrade pip first
        print_info("Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL)
        print_success("pip upgraded successfully")
        
        # Install requirements
        print_info("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_success("All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {str(e)}")
        print_warning("Try installing manually: pip install -r requirements.txt")
        return False


def verify_installations() -> Tuple[List[str], List[str]]:
    """Verify that required packages are installed"""
    print_header("Verifying Package Installations")
    
    required_packages = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("pandas", "pd"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("nibabel", "nib"),
        ("nilearn", "nilearn"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "sns"),
        ("tqdm", "tqdm"),
        ("yaml", "yaml"),
        ("networkx", "nx"),
    ]
    
    installed = []
    missing = []
    
    for package_name, import_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print_success(f"{package_name:20s} {version}")
            installed.append(package_name)
        except ImportError:
            print_error(f"{package_name:20s} NOT INSTALLED")
            missing.append(package_name)
    
    return installed, missing


def check_pytorch_device() -> str:
    """Check available PyTorch compute device"""
    print_header("Checking PyTorch Device")
    
    try:
        import torch
        
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"CUDA available: {gpu_name}")
            print_info(f"PyTorch version: {torch.__version__}")
            print_info(f"CUDA version: {torch.version.cuda}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print_success("MPS (Apple Silicon) available")
            print_info(f"PyTorch version: {torch.__version__}")
        else:
            device = "cpu"
            print_warning("No GPU detected - will use CPU")
            print_info(f"PyTorch version: {torch.__version__}")
            print_info("Training will be slower on CPU")
        
        return device
        
    except ImportError:
        print_error("PyTorch not installed!")
        return "unknown"


def create_sample_config():
    """Check if config.yaml exists"""
    print_header("Checking Configuration File")
    
    if Path("config.yaml").exists():
        print_success("config.yaml found")
        return True
    else:
        print_warning("config.yaml not found")
        print_info("Please ensure config.yaml is in the project root")
        return False


def print_next_steps(device: str):
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}✓ Environment setup successful!{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}📋 Next Steps:{Colors.ENDC}\n")
    
    print(f"{Colors.OKCYAN}1. Review Configuration{Colors.ENDC}")
    print("   Edit config.yaml to customize settings:")
    print("   - Data paths and parameters")
    print("   - Model architecture")
    print("   - Training hyperparameters")
    print()
    
    if device == "cpu":
        print(f"{Colors.WARNING}2. CPU Training Optimizations{Colors.ENDC}")
        print("   Your system will use CPU for training. Consider:")
        print("   - Reducing batch_size to 4-8 in config.yaml")
        print("   - Reducing num_epochs to 50-100")
        print("   - Using smaller synthetic dataset (n_synthetic_subjects: 100)")
        print()
    
    print(f"{Colors.OKCYAN}3. Train the Model{Colors.ENDC}")
    print("   Run training with synthetic data:")
    print(f"   {Colors.BOLD}python train.py{Colors.ENDC}")
    print()
    
    print(f"{Colors.OKCYAN}4. Evaluate Results{Colors.ENDC}")
    print("   After training completes:")
    print(f"   {Colors.BOLD}python test.py{Colors.ENDC}")
    print()
    
    print(f"{Colors.OKCYAN}5. Make Predictions{Colors.ENDC}")
    print("   Run inference on new subjects:")
    print(f"   {Colors.BOLD}python inference.py --checkpoint outputs/checkpoints/best_model.pth{Colors.ENDC}")
    print()
    
    print(f"{Colors.BOLD}📚 Documentation:{Colors.ENDC}")
    print("   - See README.md for detailed instructions")
    print("   - Check outputs/logs/ for training logs")
    print("   - View results in outputs/figures/")
    print()
    
    print(f"{Colors.BOLD}❓ Need Help?{Colors.ENDC}")
    print("   - Review troubleshooting section in README.md")
    print("   - Check configuration in config.yaml")
    print("   - Examine logs in outputs/logs/")
    print()
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}Happy training! 🚀{Colors.ENDC}\n")


def main():
    """Main setup function"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║       NEURODEVELOPMENTAL OUTCOME PREDICTOR                        ║")
    print("║       Environment Setup                                           ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    # Track success/failure
    all_checks_passed = True
    
    # 1. Check Python version
    if not check_python_version():
        all_checks_passed = False
        sys.exit(1)
    
    # 2. Create directory structure
    if not create_directory_structure():
        all_checks_passed = False
    
    # 3. Check config file
    if not create_sample_config():
        all_checks_passed = False
    
    # 4. Install dependencies
    print_info("Do you want to install dependencies now? (y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        if not install_dependencies():
            all_checks_passed = False
    else:
        print_warning("Skipping dependency installation")
        print_info("Install manually later: pip install -r requirements.txt")
    
    # 5. Verify installations
    installed, missing = verify_installations()
    
    if missing:
        print_warning(f"\n{len(missing)} packages are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print_info("Install missing packages: pip install -r requirements.txt")
        all_checks_passed = False
    
    # 6. Check PyTorch device
    device = check_pytorch_device()
    
    # 7. Print next steps
    if all_checks_passed:
        print_next_steps(device)
    else:
        print_warning("\nSetup completed with warnings")
        print_info("Please resolve the issues above before proceeding")


if __name__ == "__main__":
    main()
