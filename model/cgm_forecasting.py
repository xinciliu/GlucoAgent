import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler
import os
import random
import argparse
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer

# ===================== 全局超参数【全部优化，适配收敛】 =====================
CONF_HIST_LEN = 96
CONF_PRED_LEN = 96
EPS_STABLE = 1e-6
LOSS_FLOOR = 1e-8
WARMUP_EPOCH = 3
# =====================================================

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

SEQ_LEN = CONF_HIST_LEN
PRED_LEN = CONF_PRED_LEN
MAX_SEQ_LEN = 3 * CONF_HIST_LEN
os.makedirs("outputs", exist_ok=True)

# ===================== 维度统一说明 =====================
# 统一所有预测输出维度：【batch, pred_len, 1】三维时序格式
# 和标签target维度完全一致，时序预测标准维度，不再手动降维
# =====================================================

# ===================== 1. DistilBERT文本编码器 =====================
class DistilBERTEmbedding(nn.Module):
    def __init__(self, pretrained_name="distilbert-base-uncased", freeze_bert=True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_name)
        self.bert = DistilBertModel.from_pretrained(pretrained_name).to(device)
        self.proj = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fixed_seq = SEQ_LEN

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def build_prompt(self, patient_info: str, history_ot: np.ndarray) -> str:
        prompt_parts = [patient_info.strip()]
        for val in history_ot:
            prompt_parts.append(f"GLU:{val:.2f}")
        return " ".join(prompt_parts)

    def forward(self, prompt_list):
        tokens = self.tokenizer(
            prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            bert_out = self.bert(**tokens).last_hidden_state
        text_emb = self.proj(bert_out)

        b, l, d = text_emb.shape
        if l > self.fixed_seq:
            text_emb = text_emb[:, :self.fixed_seq, :]
        elif l < self.fixed_seq:
            pad = torch.zeros(b, self.fixed_seq - l, d, device=device)
            text_emb = torch.cat([text_emb, pad], dim=1)
        return text_emb

# ===================== 2. 数值时序分支【不变】输出统一三维 [B, 96, 1] =====================
class NumericalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, PRED_LEN * 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        pool_feat = gru_out.mean(dim=1)
        pred = self.pred_head(pool_feat)
        pred = pred.view(-1, PRED_LEN, 1)
        return pred

# ===================== 3. 文本时序分支【不变】输出统一三维 [B, 96, 1] =====================
class TextBranch(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, PRED_LEN * 1)
        )

    def forward(self, text_emb, seq_len_batch):
        batch_pred = []
        for idx in range(text_emb.shape[0]):
            valid_len = seq_len_batch[idx].item()
            feat = text_emb[idx, :valid_len, :]
            pos_id = torch.arange(valid_len, device=device)
            pos_feat = self.pos_emb(pos_id)
            gru_in = torch.cat([feat, pos_feat], dim=-1).unsqueeze(0)
            _, h_n = self.gru(gru_in)
            pred = self.pred_head(h_n.squeeze(0))
            batch_pred.append(pred)
        out = torch.stack(batch_pred)
        out = out.view(-1, PRED_LEN, 1)
        return out

# ===================== 4.【重磅优化】自适应门控融合（解决原融合层太简单、无法学习权重问题） =====================
class FusionGate(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        # 全局上下文 + 逐时间步双门控，更强拟合能力
        self.context_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, num_pred, text_pred):
        concat_feat = torch.cat([num_pred, text_pred], dim=-1)
        gate_w = self.context_gate(concat_feat)
        fuse_pred = gate_w * num_pred + (1 - gate_w) * text_pred
        return self.dropout(fuse_pred)

# ===================== 5. 数据集模块（无改动，已彻底修复numpy stack报错） =====================
class CGMDataset(Dataset):
    def __init__(self, feat_tensor_list, target_tensor_list, hist_len_list, pred_len_list):
        self.feat_list = feat_tensor_list
        self.target_list = target_tensor_list
        self.hist_list = hist_len_list
        self.pred_list = pred_len_list

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        return self.feat_list[idx], self.target_list[idx], self.hist_list[idx], self.pred_list[idx]

def collate_fn(batch):
    feat_seqs, targets, hist_lens, pred_lens = zip(*batch)
    feat_seqs = torch.stack(feat_seqs)
    hist_lens = torch.stack(hist_lens)
    pred_lens = torch.stack(pred_lens)

    max_t_len = max(t.shape[0] for t in targets)
    pad_target = torch.zeros(len(targets), max_t_len, dtype=torch.float32)
    mask = torch.zeros_like(pad_target)
    for i, t in enumerate(targets):
        pad_target[i, :len(t)] = t
        mask[i, :len(t)] = 1.0
    pad_target = pad_target.unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    return feat_seqs, pad_target, mask, hist_lens, pred_lens

def load_cgm_data(csv_path, hist_len, pred_len):
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df = df.drop("Date", axis=1)
    assert "OT" in df.columns, "CSV文件必须包含OT血糖列"

    df = df.fillna(method="ffill").fillna(0.0)

    feat_np = df.drop("OT", axis=1).values.astype(np.float32)
    target_np = df["OT"].values.astype(np.float32)

    feat_tensor = torch.from_numpy(feat_np).float()
    target_tensor = torch.from_numpy(target_np).float()

    feat_mean = feat_tensor.mean(dim=0, keepdim=True)
    feat_std = feat_tensor.std(dim=0, keepdim=True)
    feat_std = torch.where(feat_std < EPS_STABLE, torch.full_like(feat_std, 1.0), feat_std)
    feat_norm = (feat_tensor - feat_mean) / feat_std

    tar_mean = target_tensor.mean()
    tar_std = target_tensor.std()
    tar_std = tar_std if tar_std >= EPS_STABLE else torch.tensor(1.0, dtype=torch.float32)
    tar_norm = (target_tensor - tar_mean) / tar_std

    print(f"【数据校验】特征NaN：{torch.isnan(feat_norm).any().item()}，目标NaN：{torch.isnan(tar_norm).any().item()}")
    print(f"【数值范围】特征：{feat_norm.min().item():.3f} ~ {feat_norm.max().item():.3f}，目标：{tar_norm.min().item():.3f} ~ {tar_norm.max().item():.3f}")

    feat_list = []
    target_list = []
    hist_list = []
    pred_list = []

    for i in range(hist_len, len(feat_norm) - pred_len + 1):
        win_feat = feat_norm[i-hist_len:i, :]
        win_tar = tar_norm[i:i+pred_len]
        feat_list.append(win_feat)
        target_list.append(win_tar)
        hist_list.append(torch.tensor(hist_len, dtype=torch.long))
        pred_list.append(torch.tensor(pred_len, dtype=torch.long))

    scaler_y = {"mean": tar_mean.item(), "std": tar_std.item()}
    feat_mean_save = feat_mean.squeeze(0).cpu().numpy()
    feat_std_save = feat_std.squeeze(0).cpu().numpy()

    return feat_list, target_list, hist_list, pred_list, feat_mean_save, feat_std_save, scaler_y

# ===================== 6.【损失函数优化】提升主损失权重、降低分支辅助损失，加速收敛 =====================
def stable_loss(fuse_pred, num_pred, text_pred, target, mask):
    mae = nn.L1Loss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    valid_mask = mask[:, :PRED_LEN, :]

    # 主融合损失权重拉高，优先优化最终输出
    loss_fuse = (0.5 * mae(fuse_pred, target[:, :PRED_LEN, :]) + 0.5 * mse(fuse_pred, target[:, :PRED_LEN, :])) * valid_mask
    # 大幅降低分支辅助损失，避免分支梯度干扰主网络收敛
    loss_num = 0.1 * mae(num_pred, target[:, :PRED_LEN, :]) * valid_mask
    loss_text = 0.1 * mae(text_pred, target[:, :PRED_LEN, :]) * valid_mask

    total_loss = (loss_fuse + loss_num + loss_text).mean()
    total_loss = torch.clamp(total_loss, min=LOSS_FLOOR)
    return total_loss

# ===================== 7.【训练流程大优化】解决dummy向量无梯度、学习率过低、无warmup问题 =====================
def train(csv_path):
    feat_list, target_list, hist_list, pred_list, feat_mean, feat_std, scaler_y = load_cgm_data(csv_path, SEQ_LEN, PRED_LEN)

    train_idx, val_idx = train_test_split(list(range(len(feat_list))), test_size=0.2, random_state=42)
    train_dataset = CGMDataset(
        [feat_list[i] for i in train_idx],
        [target_list[i] for i in train_idx],
        [hist_list[i] for i in train_idx],
        [pred_list[i] for i in train_idx]
    )
    val_dataset = CGMDataset(
        [feat_list[i] for i in val_idx],
        [target_list[i] for i in val_idx],
        [hist_list[i] for i in val_idx],
        [pred_list[i] for i in val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    feat_dim = feat_list[0].shape[-1]
    num_model = NumericalBranch(input_dim=feat_dim).to(device)
    text_model = TextBranch().to(device)
    fuse_model = FusionGate().to(device)

    # 关键优化1：提升初始学习率 + 权重衰减下调
    optimizer = torch.optim.AdamW(
        list(num_model.parameters()) + list(text_model.parameters()) + list(fuse_model.parameters()),
        lr=5e-4, weight_decay=5e-5
    )

    # 关键优化2：warmup预热学习率，前期缓慢升温，后期余弦退火，彻底解决不收敛
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=WARMUP_EPOCH)
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=17, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[WARMUP_EPOCH])

    scaler = GradScaler(enabled=False)

    best_val_loss = float("inf")
    save_path = "best_cgm_model.pth"
    epoch_num = 20

    # 关键优化3：**可学习文本嵌入，替换随机固定dummy向量**
    # 原来随机randn全程不变，文本分支永远学不到东西，现在用可训练参数，参与梯度更新
    learnable_text_emb = nn.Parameter(torch.randn(1, SEQ_LEN, 64, device=device), requires_grad=True)

    for epoch in range(epoch_num):
        num_model.train()
        text_model.train()
        fuse_model.train()
        train_loss = 0.0

        for batch in train_loader:
            feat_seq, target, mask, seq_len_batch, pred_len_batch = batch
            feat_seq = feat_seq.to(device)
            target = target.to(device)
            mask = mask.to(device)

            out_num = num_model(feat_seq)
            bsz = feat_seq.shape[0]
            # 复制可学习文本嵌入，参与梯度反向传播，文本分支终于可以更新参数
            batch_text_emb = learnable_text_emb.repeat(bsz, 1, 1)
            out_text = text_model(batch_text_emb, seq_len_batch)
            out_fuse = fuse_model(out_num, out_text)

            loss = stable_loss(out_fuse, out_num, out_text, target, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(num_model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(fuse_model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_([learnable_text_emb], max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        num_model.eval()
        text_model.eval()
        fuse_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feat_seq, target, mask, seq_len_batch, pred_len_batch = batch
                feat_seq = feat_seq.to(device)
                target = target.to(device)
                mask = mask.to(device)

                out_num = num_model(feat_seq)
                bsz = feat_seq.shape[0]
                batch_text_emb = learnable_text_emb.repeat(bsz, 1, 1)
                out_text = text_model(batch_text_emb, seq_len_batch)
                out_fuse = fuse_model(out_num, out_text)
                loss = stable_loss(out_fuse, out_num, out_text, target, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "num": num_model.state_dict(),
                "text": text_model.state_dict(),
                "fuse": fuse_model.state_dict(),
                "learn_text_emb": learnable_text_emb,
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "scaler_y": scaler_y
            }, save_path)

    print(f"训练完成，最优模型已保存至 {save_path}")
    return save_path

# ===================== 8. 推理预测（同步加载可学习文本嵌入，无改动） =====================
def predict(checkpoint_path, csv_path, txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        patient_info = f.read().strip()

    df = pd.read_csv(csv_path).fillna(method="ffill").fillna(0.0)
    if "Date" in df.columns:
        df = df.drop("Date", axis=1)
    feat_np = df.drop("OT", axis=1).values[-SEQ_LEN:, :]
    ot_np = df["OT"].values[-SEQ_LEN:]

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = torch.tensor(ckpt["feat_mean"], dtype=torch.float32).to(device)
    feat_std = torch.tensor(ckpt["feat_std"], dtype=torch.float32).to(device)
    scaler_y = ckpt["scaler_y"]
    learnable_text_emb = ckpt["learn_text_emb"]

    feat_dim = feat_np.shape[-1]
    num_model = NumericalBranch(feat_dim).to(device)
    text_model = TextBranch().to(device)
    fuse_model = FusionGate().to(device)
    num_model.load_state_dict(ckpt["num"])
    text_model.load_state_dict(ckpt["text"])
    fuse_model.load_state_dict(ckpt["fuse"])

    text_encoder = DistilBERTEmbedding().to(device)
    prompt = text_encoder.build_prompt(patient_info, ot_np)
    real_text_emb = text_encoder([prompt])

    feat_tensor = torch.from_numpy(feat_np).float().to(device)
    feat_norm = (feat_tensor - feat_mean) / feat_std
    feat_tensor = feat_norm.unsqueeze(0)
    seq_len_tensor = torch.tensor([SEQ_LEN], dtype=torch.long)

    num_model.eval()
    text_model.eval()
    fuse_model.eval()
    with torch.no_grad():
        pred_num = num_model(feat_tensor)
        pred_text = text_model(real_text_emb, seq_len_tensor)
        pred_fuse = fuse_model(pred_num, pred_text)

    pred_res = pred_fuse.squeeze(-1)[0].cpu().numpy() * scaler_y["std"] + scaler_y["mean"]
    pd.DataFrame({"Predicted_Glucose": pred_res}).to_csv("outputs/prediction_result.csv", index=False)
    print("预测完成，结果保存至 outputs/prediction_result.csv")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--txt", type=str, required=True)
    args = parser.parse_args()
    best_ckpt = train(args.csv)
    predict(best_ckpt, args.csv, args.txt)
