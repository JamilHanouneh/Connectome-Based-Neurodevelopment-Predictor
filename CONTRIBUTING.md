# Contributing to Neurodevelopmental Outcome Predictor

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, PyTorch version)
- Error messages and stack traces

### Suggesting Enhancements

Feature requests are welcome! Please open an issue with:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if applicable)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Update documentation
7. Commit your changes (`git commit -m 'Add some feature'`)
8. Push to the branch (`git push origin feature/your-feature-name`)
9. Open a Pull Request

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Write meaningful variable names

### Testing

- Add unit tests for new features
- Ensure existing tests pass
- Test with both synthetic and real data (if available)

### Documentation

- Update README.md if needed
- Add docstrings to new code
- Update configuration documentation
- Add examples for new features

## Development Setup

Clone repository
git clone https://github.com/JamilHanouneh/neurodevelopment-predictor.git
cd neurodevelopment-predictor

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run tests
python quick_test.py


## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

## Questions?

Feel free to open an issue or contact:
- Jamil Hanouneh (jamil.hanouneh1997@gmail.com)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
