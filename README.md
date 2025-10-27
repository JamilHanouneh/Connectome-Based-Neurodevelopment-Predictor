# Connectome-Based-Neurodevelopment-Predictor

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JamilHanouneh/neurodevelopment-predictor/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/JamilHanouneh/neurodevelopment-predictor)](https://github.com/JamilHanouneh/neurodevelopment-predictor/issues)
[![GitHub stars](https://img.shields.io/github/stars/JamilHanouneh/neurodevelopment-predictor)](https://github.com/JamilHanouneh/neurodevelopment-predictor/stargazers)

**Deep Multimodal Learning for Predicting Neurodevelopmental Deficits in Very Preterm Infants**

[Getting Started](#getting-started) •
[Documentation](#documentation) •
[Citation](#citation) •
[Contributing](#contributing)

</div>

---

## Overview

This project implements a state-of-the-art deep learning system for predicting neurodevelopmental outcomes in very preterm infants using multimodal brain MRI data and clinical features. The implementation is based on the research paper by He et al. (2021) published in Frontiers in Neuroscience.

### Key Features

- **Multimodal Integration**: Combines functional connectivity (rs-fMRI), structural connectivity (DTI), anatomical features (T2w), and clinical data
- **Deep Learning Architecture**: 4-channel neural network with VGG-19 transfer learning
- **Multi-Task Learning**: Simultaneous classification (high-risk vs. low-risk) and score regression
- **Interpretability**: Grad-CAM visualization for understanding model predictions
- **CPU-Optimized**: Adapted for training on standard hardware without GPU
- **Synthetic Data Support**: Generate realistic synthetic data for testing and development
- **Production-Ready**: Complete pipeline with logging, checkpointing, early stopping, and comprehensive evaluation

---

## Clinical Application

### Problem
Very preterm infants (born ≤32 weeks gestational age) face significant risks of neurodevelopmental impairments affecting cognitive, language, and motor skills. Current clinical assessments cannot accurately predict which infants will develop deficits until age 3-5 years, missing the critical early intervention window.

### Solution
This system provides early prediction at term-equivalent age (~40 weeks postmenstrual age) by analyzing:
- Brain functional connectivity patterns (how brain regions communicate)
- Brain structural connectivity (white matter pathways)
- White matter abnormalities (diffuse injury markers)
- Clinical risk factors (birth history, maternal factors, complications)

### Expected Performance

| Outcome | Accuracy | AUC | Score Correlation (r) |
|---------|----------|-----|----------------------|
| Cognitive | 88.4% | 0.87 | 0.62 |
| Language | 87.2% | 0.85 | 0.63 |
| Motor | 86.7% | 0.85 | 0.63 |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- ~2GB disk space for dependencies

### Installation

#### Quick Setup (Automated)

```
# Clone the repository
git clone https://github.com/JamilHanouneh/neurodevelopment-predictor.git
cd neurodevelopment-predictor

# Run automated setup
python setup_environment.py
```

#### Manual Setup

```
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test (5 Minutes)

Test the entire project without downloading any dataset:

```
# Validate all components
python quick_test.py

# Run mini training (2 epochs)
python mini_train_test.py
```

---

## Dataset

### Overview

This project uses the **Developing Human Connectome Project (dHCP)** neonatal brain MRI dataset, which provides multimodal neuroimaging data from neonates and infants.

**Dataset Details:**
- **Name**: dHCP Neonatal Data Release
- **Size**: Variable (50-200GB depending on subjects selected)
- **Subjects**: 783 neonatal scans
- **Age Range**: 20-44 weeks post-conceptional age
- **Modalities**: T1w, T2w, Diffusion MRI (dMRI), Resting-state fMRI (rs-fMRI)
- **Format**: NIfTI (.nii.gz)
- **License**: Open access with data sharing agreement
- **Citation**: Edwards et al. (2022). The Developing Human Connectome Project Neonatal Data Release. Frontiers in Neuroscience.

### Why dHCP Dataset?

The dHCP dataset is ideal for this project because it provides:
- High-quality multimodal MRI data (T2w, DTI, rs-fMRI)
- Standardized preprocessing pipelines
- Large cohort of preterm and term-born infants
- Open access with proper data sharing agreements
- Well-documented data structure (BIDS-like format)

---

## Downloading the Dataset

### Prerequisites

- **Storage**: At least 200GB free disk space (for full dataset) or 50GB (for subset)
- **Internet**: Stable connection for large file downloads
- **Time**: Several hours depending on your connection speed

### Step 1: Register for Data Access

1. Visit the dHCP data portal: **https://biomedia.github.io/dHCP-release-notes/**

2. Click on **"Data Access"** or **"Download"** button

3. You will be redirected to the **NITRC Image Repository** or **XNAT Central**

4. Create a free account:
   - Go to: https://central.xnat.org/
   - Click "Register" and fill in your details
   - Verify your email address

5. Accept the **Data Use Agreement**:
   - Review the terms (academic use, no redistribution, proper citation)
   - Sign electronically

6. Request access to the dHCP project:
   - Navigate to the dHCP project page
   - Submit access request (usually approved within 1-2 business days)

### Step 2: Download the Data

You have several download options:

#### Option A: Download via Web Interface (Recommended for Beginners)

1. Log in to the XNAT Central portal

2. Navigate to: **Projects → dHCP → Experiments**

3. Select subjects you want to download:
   - For testing: Download **5-10 subjects** (~10-20GB)
   - For full training: Download **50-100 subjects** (~100-200GB)
   - **Tip**: Prioritize subjects with complete data (T2w + dMRI + rs-fMRI)

4. For each subject:
   - Click on the subject ID (e.g., `sub-CC00050XX01`)
   - Go to **"Actions"** → **"Download Images"**
   - Select modalities:
     - ✓ Anatomical (T2w required, T1w optional)
     - ✓ Diffusion (dMRI/DTI)
     - ✓ Functional (rs-fMRI)
   - Choose format: **NIfTI** (.nii.gz)
   - Click **"Download"**

5. Repeat for all desired subjects

#### Option B: Bulk Download via Command Line (Advanced Users)

```
# Install XNAT downloader
pip install xnat

# Use the download script (create this file)
python scripts/download_dhcp_data.py --username YOUR_USERNAME --subjects 10
```

#### Option C: Download Specific Files

If you only want to test with minimal data:

1. Download at least **3-5 subjects** with complete scans
2. Minimum required files per subject:
   - `*_T2w.nii.gz` (anatomical)
   - `*_dwi.nii.gz`, `*_dwi.bval`, `*_dwi.bvec` (diffusion)
   - `*_bold.nii.gz` (functional MRI)

### Step 3: Organize Downloaded Data

After downloading, organize the data in the following structure:

```
neurodevelopment_predictor/
└── data/
    └── raw/
        ├── sub-CC00050XX01/
        │   └── ses-001/
        │       ├── anat/
        │       │   ├── sub-CC00050XX01_ses-001_T2w.nii.gz
        │       │   └── sub-CC00050XX01_ses-001_T1w.nii.gz  (optional)
        │       ├── dwi/
        │       │   ├── sub-CC00050XX01_ses-001_dwi.nii.gz
        │       │   ├── sub-CC00050XX01_ses-001_dwi.bval
        │       │   └── sub-CC00050XX01_ses-001_dwi.bvec
        │       └── func/
        │           └── sub-CC00050XX01_ses-001_task-rest_bold.nii.gz
        ├── sub-CC00050XX02/
        │   └── ses-001/
        │       └── ...
        └── participants.csv  (create this - see below)
```

#### Detailed Organization Steps:

1. **Extract downloaded ZIP files**:
   ```
   # Navigate to download location
   cd ~/Downloads
   
   # Extract all ZIP files
   unzip "*.zip" -d extracted_data/
   ```

2. **Move to project directory**:
   ```
   # Create data directories
   mkdir -p data/raw
   
   # Move extracted subjects
   mv extracted_data/sub-* data/raw/
   ```

3. **Verify file structure**:
   ```
   # Check structure
   ls -R data/raw/sub-CC00050XX01/
   
   # Should show:
   # ses-001/anat/
   # ses-001/dwi/
   # ses-001/func/
   ```

4. **Create participants metadata file**:
   Create `data/raw/participants.csv` with subject information:
   
   ```
   subject_id,age_at_scan_weeks,gestational_age_weeks,birth_weight_g,sex
   sub-CC00050XX01,42.5,28.0,1150,M
   sub-CC00050XX02,41.2,29.5,1320,F
   sub-CC00050XX03,40.8,27.2,980,M
   ...
   ```
   
   **Note**: You can extract this information from the dHCP metadata files or create synthetic clinical data for testing.

### Step 4: Update Configuration

Once data is organized, update `config.yaml`:

```
data:
  mode: "real"  # Change from "synthetic"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
```

### Step 5: Verify Data Integrity

Run the data validation script:

```
python scripts/validate_data.py
```

This will check:
- All required files are present
- File formats are correct
- No corrupted files
- Subjects have complete modalities

---

## Alternative: Start with Synthetic Data

If you want to test the pipeline immediately without downloading the full dataset:

1. **No download required** - Synthetic data is generated automatically

2. **Configure for synthetic mode**:
   ```
   data:
     mode: "synthetic"
     n_synthetic_subjects: 150
   ```

3. **Run training**:
   ```
   python train.py
   ```

The system will automatically generate realistic synthetic:
- Functional connectivity matrices (223×223)
- Structural connectivity matrices (90×90)
- DWMA features (11 measurements)
- Clinical features (72 variables)
- Outcome labels (Bayley-III scores)

**When to use synthetic data:**
- Testing the pipeline
- Development and debugging
- Learning the codebase
- Demonstrations

**When to use real dHCP data:**
- Production training
- Research publications
- Clinical validation
- Performance benchmarking

---

## Dataset Citation

If you use the dHCP dataset, please cite:

```
@article{edwards2022dhcp,
  title={The Developing Human Connectome Project Neonatal Data Release},
  author={Edwards, A Dale and Rueckert, Daniel and Smith, Stephen M and 
          Abo Seada, Suzan and Alansary, Amir and Arichi, Tomoki and 
          Bastiani, Matteo and others},
  journal={Frontiers in Neuroscience},
  volume={16},
  pages={886772},
  year={2022},
  publisher={Frontiers Media SA},
  doi={10.3389/fnins.2022.886772}
}
```

---

## Data Usage Notes

**Important Considerations:**

1. **Ethics and Privacy**:
   - The dHCP data is de-identified
   - Follow institutional IRB guidelines
   - Do not attempt to re-identify subjects

2. **Data Sharing**:
   - Do NOT redistribute raw dHCP data
   - Others should download from official sources
   - You may share processed features (with permission)

3. **Storage Requirements**:
   - Raw data: ~2-5GB per subject
   - Processed data: ~50-100MB per subject
   - Model outputs: ~1-2GB

4. **Processing Time**:
   - Data download: 2-6 hours (full dataset)
   - Preprocessing: 5-10 minutes per subject
   - Training: 4-8 hours (CPU) or 1-2 hours (GPU)

---

## Troubleshooting Dataset Issues

### "Cannot access dHCP data"
- Ensure your access request is approved (check email)
- Verify you're logged into XNAT Central
- Try a different browser or clear cache

### "Files are corrupted"
- Re-download the affected subject
- Verify file integrity with checksums (if provided)
- Check available disk space

### "Missing modalities"
- Some subjects may not have all modalities
- Use the data validation script to identify complete subjects
- Minimum requirement: T2w + one of (dMRI or rs-fMRI)

### "File structure doesn't match"
- dHCP uses BIDS-like structure
- Ensure you extract to the correct directory
- Run the organization script (if provided)

### "Download is too slow"
- Try downloading during off-peak hours
- Use a download manager for large files
- Consider downloading a subset first for testing

For additional help:
- dHCP documentation: https://biomedia.github.io/dHCP-release-notes/
- Open an issue: https://github.com/JamilHanouneh/Multimodal-NeuroPredict/issues
- Contact: jamil.hanouneh1997@gmail.com
```

### Training

Train the model with synthetic data (no dataset download required):

```
python train.py
```

Train with custom parameters:

```
python train.py --epochs 100 --batch_size 8 --lr 0.0001
```

### Evaluation

Evaluate trained model on test set:

```
python test.py --checkpoint outputs/checkpoints/best_model.pth
```

### Inference

Make predictions on new subjects:

```
python inference.py --checkpoint outputs/checkpoints/best_model.pth \
                     --input_dir data/new_subjects \
                     --output_dir outputs/predictions
```

---

## Project Structure

```
neurodevelopment_predictor/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── setup_environment.py        # Environment setup
├── train.py                    # Training script
├── test.py                     # Testing script
├── inference.py                # Inference script
├── quick_test.py              # Quick validation test
├── mini_train_test.py         # Mini training test
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Neural network models
│   ├── training/              # Training utilities
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Utility functions
├── outputs/                   # Output directory
│   ├── checkpoints/           # Model checkpoints
│   ├── logs/                  # Training logs
│   ├── predictions/           # Predictions
│   └── figures/               # Visualizations
└── notebooks/                 # Jupyter notebooks
```

---

## Scientific Background

### Multimodal MRI Data

#### Functional Connectivity (rs-fMRI)
- 223×223 correlation matrix representing functional interactions between brain regions
- Captures resting-state brain network organization
- Reflects neural communication efficiency

#### Structural Connectivity (DTI)
- 90×90 fiber count matrix showing white matter pathways
- Quantifies physical connections via diffusion tractography
- Indicates structural brain wiring integrity

#### Diffuse White Matter Abnormality (DWMA)
- 11 volumetric features from T2-weighted MRI
- Quantifies white matter injury
- Strong predictor of motor and cognitive outcomes

#### Clinical Features
- 72 perinatal variables including birth characteristics, maternal factors, medical complications, and socioeconomic factors

---

## Configuration

Edit `config.yaml` to customize:

```
# Model architecture
model:
  pretrained_vgg19: true
  fc_dims: 
  dropout_rate: 0.5

# Training parameters
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.0001
```

---

## Results and Outputs

### Training Outputs
- Model checkpoints: `outputs/checkpoints/best_model.pth`
- Training logs: `outputs/logs/training_*.log`
- Loss curves: `outputs/figures/loss_curves.png`

### Evaluation Outputs
- Performance metrics: `outputs/predictions/test_metrics.csv`
- ROC curves: `outputs/figures/roc_curves.png`
- Confusion matrices: `outputs/figures/confusion_matrices.png`
- Scatter plots: `outputs/figures/prediction_scatter.png`
- HTML report: `outputs/reports/evaluation_report.html`

---

## Citation

If you use this code in your research, please cite:

### This Implementation
```
@software{hanouneh2025neurodevelopment,
  author = {Hanouneh, Jamil},
  title = {Neurodevelopmental Outcome Predictor: Deep Multimodal Learning for Preterm Infants},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JamilHanouneh/neurodevelopment-predictor}
}
```

### Original Paper
```
@article{he2021deep,
  title={Deep Multimodal Learning From MRI and Clinical Data for Early Prediction of Neurodevelopmental Deficits in Very Preterm Infants},
  author={He, Lili and Li, Hailong and Chen, Ming and Wang, Jinghua and Altaye, Mekibib and Dillman, Jonathan R and Parikh, Nehal A},
  journal={Frontiers in Neuroscience},
  volume={15},
  pages={753033},
  year={2021},
  publisher={Frontiers Media SA},
  doi={10.3389/fnins.2021.753033}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Original Research**: He et al. (2021), Cincinnati Children's Hospital Medical Center
- **dHCP Dataset**: Developing Human Connectome Project consortium
- **Pre-trained Models**: VGG-19 from ImageNet (Simonyan & Zisserman, 2014)
- **Neuroimaging Tools**: nilearn, nibabel, scikit-image communities

---

## Contact

**Jamil Hanouneh**
- Email: jamil.hanouneh1997@gmail.com
- GitHub: [JamilHanouneh](https://github.com/JamilHanouneh)
- Affiliation: [Friedrich-Alexander-Universität Erlangen-Nürnberg](https://www.fau.eu/)

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact via email.

---
