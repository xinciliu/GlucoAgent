import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealing
from torch.amp import GradScaler, autocast
import os
import json
import random
import argparse
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer

# ===================== Global Fixed Configuration =====================
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set computing device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Fixed sequence parameters: input length and 24-hour prediction length
SEQ_LEN = 96
PRED_LEN_24H = 96

# Create output directory if it does not exist
os.makedirs("outputs", exist_ok=True)

# ===================== 1. DistilBERT Embedding Module =====================
class DistilBERTEmbedding(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased', hidden_dim=768, freeze_bert=True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model = DistilBertModel.from_pretrained(pretrained_model_name).to(device)
        self.hidden_dim = hidden_dim

# Freeze parameters of pre-trained DistilBERT
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

# Dimension projection layer: 768 -> 64
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(device)

    def build_prompt(self, patient_info: str, history_seq: np.ndarray) -> str:
        """
        Assemble prompt following rules:
        Patient basic information + ###history_point1### + ###history_point2### ...
        Total number of history points equals SEQ_LEN
        """
        prompt_parts = [patient_info.strip()]
        # Concatenate each historical data point with specified markers
        for val in history_seq:
            prompt_parts.append(f"###{val:.2f}###")
        return " ".join(prompt_parts)

    def forward(self, prompt_list):
        """
        Input: List of prompt strings
        Output: Text embedding with shape [batch_size, seq_len, 64]
        """
        inputs = self.tokenizer(
            prompt_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Disable gradient calculation for frozen BERT
        with torch.no_grad() if not self.training else torch.enable_grad():
            bert_out = self.bert_model(**inputs)
        bert_emb = bert_out.last_hidden_state  # Shape: [Batch, Length, 768]
        proj_emb = self.projection(bert_emb)  # Shape: [Batch, Length, 64]

        # Align sequence length to fixed SEQ_LEN
        b, l, d = proj_emb.shape
        if l > SEQ_LEN:
            proj_emb = proj[:, :SEQ_LEN, :]
        elif l < SEQ_LEN:
            pad = torch.zeros(b, SEQ_LEN - l, d, device=device)
            proj_emb = torch.cat([proj_emb, pad], dim=1)
        return proj_emb

# ===================== 2. Main Model Architecture =====================
class UnifiedNumericalFramework(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, max_seq_len=96*3, dropout_rate=0.3,
                 raw_threshold_x=None, raw_threshold_y=None, target_scaler=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

# Bidirectional GRU for time series modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )

# Boost module for high blood glucose features
        self.high_boost = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.Sigmoid()
        )

# Boost module for low blood glucose features
        self.low_boost = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.Sigmoid()
        )

# Threshold attention module for abnormal glucose values
        self.threshold_attn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

# Prediction heads for different prediction lengths
        self.pred_heads = nn.ModuleDict({
            f"pred_{pl}": nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, pl)
            ) for pl in {60*3,72*3,96*3}
        })

# Register threshold values as buffer
        if raw_threshold_x is not None:
            if target_scaler:
                sx = target_scaler.transform([[raw_threshold_x]])[0][0]
                self.register_buffer("threshold_x", torch.tensor(sx, dtype=torch.float32))
            else:
                self.register_buffer("threshold_x", torch.tensor(raw_threshold_x, dtype=torch.float32))
        if raw_threshold_y is not None:
            if target_scaler:
                sy = target_scaler.transform([[raw_threshold_y]])[0][0]
                self.register_buffer("threshold_y", torch.tensor(sy, dtype=torch.float32))
            else:
                self.register_buffer("threshold_y", torch.tensor(raw_threshold_y, dtype=torch.float32))

    def forward(self, x, pred_len):
        gru_out, _ = self.gru(x)
        attn_w = torch.softmax(gru_out.mean(dim=-1), dim=1).unsqueeze(-1)
        feat = (gru_out * attn_w).sum(dim=1)

# Apply threshold-aware feature enhancement
        t_w = self.threshold_attn(feat)
        h_w = self.high_boost(feat)
        feat = feat * (1 + t_w * h_w)
        if hasattr(self, "threshold_y"):
            l_w = self.low_boost(feat)
            feat = feat * (1 + t_w * l_w)

        max_pl = max(pred_len).item()
        outputs = []
        for i in range(x.size(0)):
            pl = pred_len[i].item()
            pred = self.pred_heads[f"pred_{pl}"](feat[i])
            if pred.dim() == 2:
                pred = pred.squeeze(0)
            if pl < max_pl:
                pred = nn.functional.pad(pred, (0, max_pl-pl))
            outputs.append(pred)
        return torch.stack(outputs)

class UnifiedTextualFramework(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, max_seq_len=96*3, dropout_rate=0.3,
                 raw_threshold_x=None, raw_threshold_y=None, target_scaler=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

# Positional encoding for text sequence
        self.pos_enc = nn.Embedding(max_seq_len, hidden_dim)

# GRU for text feature extraction
        self.gru = nn.GRU(input_dim + hidden_dim, hidden_dim, num_layers=1, batch_first=True)

# Sensitivity modules for high and low glucose levels
        self.sens_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.sens_low = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

# Prediction heads for different prediction lengths
        self.pred_heads = nn.ModuleDict({
            f"pred_{pl}": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, pl)
            ) for pl in {60*3,72*3,96*3}
        })

# Register threshold values as buffer
        if raw_threshold_x is not None:
            sx = target_scaler.transform([[raw_threshold_x]])[0][0] if target_scaler else raw_threshold_x
            self.register_buffer("threshold_x", torch.tensor(sx, dtype=torch.float32))
        if raw_threshold_y is not None:
            sy = target_scaler.transform([[raw_threshold_y]])[0][0] if target_scaler else raw_threshold_y
            self.register_buffer("threshold_y", torch.tensor(sy, dtype=torch.float32))

    def forward(self, x, seq_len, pred_len):
        b = x.size(0)
        max_pl = max(pred_len).item()
        outputs = []
        for i in range(b):
            sl = seq_len[i].item()
            seq = x[i, :sl]
            pos = torch.arange(sl, device=x.device)
            pos_emb = self.pos_enc(pos)
            seq = torch.cat([seq, pos_emb], dim=-1).unsqueeze(0)
            _, hn = self.gru(seq)
            hn = hn.squeeze(0)

# Apply sensitivity enhancement
            sw_h = self.sens_high(hn)
            feat = hn * (1 + sw_h)
            if hasattr(self, "threshold_y"):
                sw_l = self.sens_low(hn)
                feat = feat * (1 + sw_l)

            pl = pred_len[i].item()
            pred = self.pred_heads[f"pred_{pl}"](feat)
            if pred.dim() == 2:
                pred = pred.squeeze(0)
            if pl < max_pl:
                pred = nn.functional.pad(pred, (0, max_pl-pl))
            outputs.append(pred)
        return torch.stack(outputs)

class UnifiedGatingFusion(nn.Module):
    def __init__(self, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
# Gating modules for different prediction lengths
        self.gates = nn.ModuleDict({
            f"gate_{pl}": nn.Sequential(
                nn.Linear(pl*2, pl*2), nn.ReLU(),
                nn.Linear(pl, pl), nn.Sigmoid()
            ) for pl in {60*3,72*3,96*3}
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, num_out, text_out, pred_len):
        max_pl = max(pred_len).item()
        outputs = []
        for i in range(num_out.size(0)):
            pl = pred_len[i].item()
            n = num_out[i, :pl].flatten()
            t = text_out[i, :pl].flatten()
            comb = torch.cat([n,t])
# Adaptive gating fusion
            g = self.gates[f"gate_{pl}"](comb)
            fused = g * n + (1-g) * t
            if pl < max_pl:
                fused = nn.functional.pad(fused, (0, max_pl-pl))
            outputs.append(self.dropout(fused))
        return torch.stack(outputs)

# ===================== 3. Dataset & Loss Function =====================
class TimeSeriesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        num_seq, target, seq_len, pred_len = self.samples[idx]
        return (
            torch.tensor(num_seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(pred_len, dtype=torch.long)
        )

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    num_seqs, targets, seq_lens, pred_lens = zip(*batch)
    num_seqs = torch.stack(num_seqs, 0)
    seq_lens = torch.stack(seq_lens, 0)
    pred_lens = torch.stack(pred_lens, 0)

    max_t = max(t.size(0) for t in targets)
    pad_target = torch.zeros(len(targets), max_t, dtype=torch.float32)
    mask = torch.zeros_like(pad_target)
    for i, t in enumerate(targets):
        pad_target[i,:t.size(0)] = t
        mask[i,:t.size(0)] = 1.0
    return num_seqs, pad_target, mask, seq_lens, pred_lens

def load_raw_data(csv_path, seq_len, pred_len):
    """Load and preprocess CGM time series data"""
    df = pd.read_csv(csv_path)
    assert "OT" in df.columns, "CSV must contain column 'OT' (ground truth glucose)"
    feats = df.drop("OT", axis=1).values
    label = df["OT"].values
    assert len(df) > seq_len + pred_len, "Dataset length is insufficient"

# Standardization
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    feats = scaler_x.fit_transform(feats)
    label = scaler_y.fit_transform(label.reshape(-1,1)).flatten()

# Generate sliding window samples
    samples = []
    for t in range(seq_len, len(feats)-pred_len + 1):
        s = feats[t-seq_len:t]
        y = label[t:t+pred_len]
        samples.append((s, y, seq_len, pred_len))
    return samples, scaler_x, scaler_y

def threshold_loss(preds, target, mask, pred_len, scaler_y, beta=4.0):
    """Combined loss with threshold awareness and knowledge distillation"""
    mae = nn.L1Loss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    total = 0.0
    b = preds[0].size(0)
    for i in range(b):
        pl = pred_len[i].item()
        p0 = preds[0][i,:pl]
        y = target[i,:pl]
        msk = mask[i,:pl]
# Mixed MAE and MSE base loss
        base = (0.7*mae(p0,y) + 0.3*mse(p0,y)) * msk
        base = base.mean()

        w = 1.0
        total += base
# Knowledge distillation loss between different branches
        for j in range(1, len(preds)):
            pd = preds[j][i,:pl]
            dist = mse(p0, pd) * msk
            loss_d = 0.7*mae(pd,y) + 0.3*mse(pd,y)
            total += 0.5*dist.mean() + 0.5*loss_d.mean()
    return total / b

def post_process(pred, pred_len, scaler_y):
    """Post-process prediction output"""
    out = []
    max_pl = max(pred_len).item()
    for i in range(pred.size(0)):
        pl = pred_len[i].item()
        p = pred[i,:pl]
        out.append(p)
    return torch.stack(out)

# ===================== 4. Training Pipeline =====================
def train_pipeline(csv_path):
    """Train model with full dataset and save weights"""
    print("=== Start Training ===")
# Load dataset and split train / validation set
    samples, _, scaler_y = load_raw_data(csv_path, SEQ_LEN, PRED_LEN_24H)
    train_samp, val_samp = train_test_split(samples, test=0.2, random_state=42)

    train_ds = TimeSeriesDataset(train_samp)
    val_ds = TimeSeriesDataset(val_samp)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize all model components
    dummy_seq = train_samp[0][0]
    in_dim = dummy_seq.shape[-1]
    bert_emb = DistilBERTEmbedding()
    num_model = UnifiedNumericalFramework(input_dim=in_dim).to(device)
    text_model = UnifiedTextualFramework(input_dim=64).to(device)
    fuse_model = UnifiedGatingFusion().to(device)
    model_dict = {"num":num_model, "text":text_model, "fuse":fuse_model}

# Optimizer and learning rate scheduler
    params = list(num_model.parameters()) + list(text_model.parameters()) + list(fuse_model.parameters())
    opt = torch.optim.AdamW(params, lr=0.01, weight_decay=1e-3)
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5*len(train_loader))
    cos = CosineAnnealingLR(opt, T_max=15*len(train_loader), eta_min=1e-6)
    sched = SequentialLR(opt, schedulers=[warmup, cos], milestones=[5*len(train_loader)])
    scaler = GradScaler(enabled=device.type=="cuda")

    best_loss = float("inf")
    save_path = "best_model.pth"
    epochs = 20
    patience = 50

# Training loop
    for ep in range(epochs):
        for m in model_dict.values():
            m.train()
        tr_loss = 0.0
        for batch in train_loader:
            num_seq, target, mask, sl, pl = batch
            num_seq = num_seq.to(device)
            target = target.to(device)
            mask = mask.to(device)

            with autocast(enabled=device.type=="cuda"):
                o_num = num_model(num_seq, pl)
# Dummy text feature for training phase
                dummy_text = torch.randn_like(num_seq[:,:,:64])
                o_text = text_model(dummy_text, sl, pl)
                o_fuse = fuse_model(o_num, o_text, pl)
                all_out = [o_fuse, o_num, o_text]
                loss = threshold_loss(all_out, target, mask, pl, scaler_y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

# Validation loop
        for m in model_dict.values():
            m.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                num_seq, target, mask, sl, pl = batch
                num_seq = num_seq.to(device)
                target = target.to(device)
                mask = mask.to(device)
                o_num = num_model(num_seq, pl)
                dummy_text = torch.randn_like(num_seq[:,:,:64])
                o_text = text_model(dummy_text, sl, pl)
                o_fuse = fuse_model(o_num, o_text, pl)
                all_out = [o_fuse, o_num, o_text]
                loss = threshold_loss(all_out, target, mask, pl, scaler_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {ep+1:2d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

# Save best model and early stop mechanism
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "num": num_model.state_dict(),
                "text": text_model.state_dict(),
                "fuse": fuse_model.state_dict(),
                "scaler_y": scaler_y,
                "cfg": {"seq_len":SEQ_LEN, "pred_len":PRED_LEN_24H}
            }, save_path)
            patience = 50
        else:
            patience -= 1
            if patience <= 0:
                print("Early stop!")
                break
    print(f"Training finished. Best model saved to {save_path}")
    return save_path, scaler_y

# ===================== 5. Inference for 24-hour Prediction =====================
def predict_24h(model_path, csv_path, patient_txt_path):
    """
    Inference pipeline:
    Use the last complete sequence from dataset as input,
    combine patient info and history to build prompt, then predict future 24h glucose
    """
    print("=== Start 24h Prediction ===")
# Read patient information text file
    with open(patient_txt_path, "r", encoding="utf-8") as f:
        patient_info = f.read().strip()

# Load raw data and extract the last complete input sequence
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    feats_raw = df.drop("OT", axis=1).values
    label_raw = df["OT"].values
    last_seq = feats_raw[-SEQ_LEN:]
    last_label_history = label_raw[-SEQ_LEN:]

# Build standard prompt with patient info and historical data markers
    bert = DistilBERTEmbedding().to(device)
    prompt = bert.build_prompt(patient_info, last_label_history)
    prompt_list = [prompt]

# Load trained model and standardization tool
    ckpt = torch.load(model_path, map_location=device)
    scaler_y = ckpt["scaler_y"]
    in_dim = last_seq.shape[-1]
    num_model = UnifiedNumericalFramework(input_dim=in_dim).to(device)
    text_model = UnifiedTextualFramework(input_dim=64).to(device)
    fuse_model = UnifiedGatingFusion().to(device)
    num_model.load_state_dict(ckpt["num"])
    text_model.load_state_dict(ckpt["text"])
    fuse_model.load_state_dict(ckpt["num"])

# Standardize input sequence
    _, scaler_x = load_raw_data(csv_path, SEQ_LEN, PRED_LEN_24H)
    seq_norm = scaler_x.transform(last_seq)
    seq_tensor = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(device)
    pred_len_tensor = torch.tensor([PRED_LEN_24H], dtype=torch.long)
    seq_len_tensor = torch.tensor([SEQ_LEN], dtype=torch.long)

# Encode prompt via DistilBERT
    text_emb = bert(prompt_list)

# Model inference
    num_model.eval()
    text_model.eval()
    fuse_model.eval()
    with torch.no_grad():
        out_num = num_model(seq_tensor, pred_len_tensor)
        out_text = text_model(text_emb, seq_len_tensor, pred_len_tensor)
        out_fuse = fuse_model(out_num, out_text, pred_len_tensor)
        pred_norm = post_process(out_fuse, pred_len_tensor, scaler_y)

# Inverse standardization to get real glucose values
    pred_np = pred_norm[0].cpu().numpy()
    pred_real = scaler_y.inverse_transform(pred_np.reshape(-1,1)).flatten()

# Save prediction results
    res_df = pd.DataFrame({
        "Predicted_24h_Glucose": pred_real
    })
    res_path = os.path.join("outputs", "24h_glucose_prediction.csv")
    res_df.to_csv(res_path, index=False)
    print(f"24-hour prediction completed. Results saved to: {res_path}")
    return pred_real

# ===================== 6. Command Line Entry =====================
def main():
    parser = argparse.ArgumentParser(description="GlucoAgent: CGM 24h Prediction")
    parser.add_argument("--csv", required=True, help="Path to CGM time series csv file (contain 'OT' column)")
    parser.add_argument("--txt", required=True, help="Path to patient information txt file")
    args = parser.parse()

# Execute training and prediction workflow
    model_file, _ = train_pipeline(args.csv)
    predict_24h(model_file, args.csv, args.txt)

if __name__ == "__main__":
    main()
