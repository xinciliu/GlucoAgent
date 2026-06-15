# GlucoAgent: Multimodal AI for Long-Term Glucose Prediction
## Project Overview
This repository implements the core model of **GlucoAgent**, a multimodal artificial intelligence framework proposed in the paper *GlucoAgent: A Multimodal AI Framework for Long-Term Glucose Prediction and Personalized Diabetes Management*.

The framework is designed for long-term continuous glucose monitoring (CGM) prediction, targeted at diabetes management scenarios. It combines time-series numerical features and automatically generated text embeddings to achieve accurate glucose forecasting, early detection of hyperglycemia and hypoglycemia, and supports personalized model adaptation with limited patient data.

## Key Features
1. **Dual-Branch Multimodal Architecture**
   Build separate encoding branches for numerical CGM time-series and text semantics, fully mining multi-dimensional information from raw monitoring data.
2. **Automatic Text Embedding Generation**
   Integrated DistilBERT to convert numerical time-series data into natural language descriptions and generate text embeddings automatically. **No manual production of text CSV files is required**, only raw numerical data is needed.
3. **Enhanced Physiological Perception Module**
   Equipped with glycemia refinement components and threshold-aware loss functions to strengthen the model’s sensitivity to clinically critical hyperglycemia and hypoglycemia events.
4. **Gated Fusion & Knowledge Distillation**
   Adopt adaptive gated fusion to balance contributions from numerical and text branches. Combined with knowledge distillation to improve model generalization and prediction stability.
5. **Data-Efficient Fine-Tuning**
   Support uniform tuning strategy, which enables rapid model personalization for new users or scenarios with limited historical CGM data.
6. **Multi-Step Long-Term Prediction**
   Support multi-task long-term glucose prediction with different prediction horizons, covering 15h, 18h and 24h forecasting requirements in practical diabetes management.

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
CGM raw numerical CSV file is required: `cgm_ts.csv`

### File Format Requirements
- The CSV file contains multiple feature columns for CGM monitoring (glucose value, carbohydrate intake, insulin dosage).

- A fixed target column named **OT** must be included, which represents the ground-truth blood glucose value for model prediction.

## Project Structure
