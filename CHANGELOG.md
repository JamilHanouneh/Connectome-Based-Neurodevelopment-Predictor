# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-26

### Added
- Initial release of Neurodevelopmental Outcome Predictor
- 4-channel multimodal deep learning architecture
- VGG-19 based feature extraction for connectivity matrices
- Support for functional connectivity (rs-fMRI) analysis
- Support for structural connectivity (DTI) analysis
- DWMA feature extraction module
- Clinical data processing pipeline
- Multi-task learning (classification + regression)
- Synthetic data generation for testing
- Complete training pipeline with early stopping
- Comprehensive evaluation metrics (classification + regression)
- Grad-CAM visualization for interpretability
- HTML report generation
- Automated environment setup script
- Quick validation test suite
- Mini training test for rapid validation
- Extensive documentation and README

### Features
- Predicts cognitive, language, and motor outcomes
- CPU-optimized training
- Configurable via YAML
- Production-ready logging and checkpointing
- ROC curves, confusion matrices, scatter plots
- Bland-Altman analysis
- Command-line interface for train/test/inference

## [Unreleased]

### Planned
- Multi-center validation
- Integration with BIDS format
- Attention-based fusion mechanisms
- Longitudinal outcome prediction
- Uncertainty quantification
- Support for additional brain atlases
- GUI for clinical deployment
