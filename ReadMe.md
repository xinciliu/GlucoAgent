# GlucoAgent: Multimodal AI for Long-Term Glucose Prediction
## Project Overview
This repository implements the core model of **GlucoAgent**, a multimodal artificial intelligence framework proposed in the paper *GlucoAgent: A Multimodal AI Framework for Long-Term Glucose Prediction and Personalized Diabetes Management*.

GlucoAgent integrates system with two core functions:
1. **Long-term CGM Prediction**: Forecast future blood glucose based on CGM time series data.
2. **Personalized Recommendation**: Generate diabetes lifestyle guidance using predicted results and user profile.


## Environment & Dependencies
### Supported Python Version
Python 3.8 / 3.9 / 3.10

### Required Libraries
All dependencies are listed in `requirements.txt`, including:
- PyTorch: Deep learning framework for model construction and training
- Transformers: Load pre-trained DistilBERT for text embedding
- Pandas & NumPy: Raw data loading, processing and numerical computation
- Scikit-learn: Data normalization, dataset splitting and evaluation
- Matplotlib (Optional): Data visualization

### Hardware Recommendation
- **GPU (Recommended)**: NVIDIA GPU with CUDA support for mixed precision training and acceleration
- **CPU**: Supported for operation, but with slower training and inference speed

## Data Specification
### Data File
`./data/cgm_ts.csv`, which contains CGM time-series features and a mandatory column named **OT** (ground truth glucose value), as well as carbohydrate intake and insulin dosage.
`./data/user_info.txt`, Plain text for user information, including age, gender, diabetes type, medical history and living habits.


## Project Structure
## Project Structure
- `main.py`: Main program (train / predict / generate recommendation)
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation
- `config.yaml`: Unified configuration file
- `data/`
  - `cgm_data.csv`: CGM time series input
  - `user_info.txt`: User description text
- `reference.json`: Reference literature for recommendation module
- `best_model.pth`: Trained model weights (auto-generated)
- `outputs/`: Prediction & recommendation results (auto-generated)
