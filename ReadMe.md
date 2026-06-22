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

`./data/reference_paper.json`, reference paper that is extracted from database that are closely related to this patient.

## Project Structure
## Project Structure
- `main.py`: Main program (train / predict / generate recommendation)
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation
- `data/`
  - `cgm_data.csv`: CGM time series input
  - `user_info.txt`: User description text
  - `reference_paper.json`: reference paper related to this patient
- `model/`
  - `cgm_forecasting.py`: CGM time series forecasting
  - `recommendation_generation.py`: Recommendation Generation
- `best_model.pth`: Trained model weights (auto-generated)
- `outputs/`: Prediction & recommendation results (auto-generated)

## Usage Pipeline

- The full execution workflow contains two sequential steps:
  - Run time-series forecasting script to generate predicted glucose CSV in outputs/ folder
  - Run LLM recommendation script using predicted glucose file, patient profile and reference papers to generate personalized advice JSON in outputs/

### Step 1: Train Model & Generate Glucose Prediction Results
- Required arguments:
- --csv: original CGM data, including OT( which means CGM value), dietary_transfer, insulin_transfer
- --txt: a paragraph that represent the basic information of patient.

Command:

- python ./model/cgm_forecasting.py --csv ./data/cgm_data.csv --txt ./data/user_info.txt

- Output:
  - Model checkpoint: best_model.pth (project root directory)
  - Predicted glucose file: outputs/prediction_result.csv

### Step 2:
Step 1 must finish successfully first, otherwise the predicted glucose file will be missing and trigger loading error.
Required arguments:

--hist_cgm: Raw original CGM time series CSV path
--pred_cgm: Predicted glucose CSV generated in Step1
--user_info: Patient personal information text file
--reference: Diabetes research reference JSON file

Command:

python ./model/recommendation_generation.py \
--hist_cgm ./data/cgm_data.csv \
--pred_cgm ./outputs/prediction_result.csv \
--user_info ./data/user_info.txt \
--reference ./data/reference_paper.json

Output:
LLM structured JSON recommendations printed in terminal
Result file: outputs/recommendation_result.json




