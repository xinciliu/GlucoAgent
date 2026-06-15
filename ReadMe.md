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
GlucoAgent/

├── main.py               # 唯一入口：训练/预测/推荐 全流程主代码

├── requirements.txt      # 依赖清单

├── README.md             # 项目说明（英文）

├── config.yaml           # 统一配置文件（模型/数据/LLM 所有参数）

├── data/                 # 输入数据文件夹

│   ├── cgm_data.csv      # CGM 时序输入（必填）

│   └── user_info.txt     # 用户描述文本（必填）

├── reference.json        # 推荐模块：糖尿病参考文献库

├── best_model.pth        # 自动生成：训练好的模型权重

└── outputs/              # 自动生成：预测结果 + 推荐文本


## Project Structure
