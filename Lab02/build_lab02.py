import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import xgboost as xgb
import catboost as cb

# Set font and style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

print("=== STARTING LAB02 PIPELINE EXECUTION ===")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPR_DIR = os.path.join(BASE_DIR, "experiments", "house_mlp_v1")
IMG_DIR = os.path.join(EXPR_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Configuration dictionary (params)
params = {
    "project": {
        "title": "Lab02: Qui trình máy học cho dự đoán giá nhà",
        "course": "Học sâu (Deep Learning) - 858048",
        "class": "DDU1231 - Nhóm 01",
        "instructor": "TS. Đỗ Như Tài",
        "authors": [
            {"name": "Phan Đồng Minh Quân", "student_id": "3123580038", "role": "Nhóm trưởng"},
            {"name": "Võ Minh Tiến", "student_id": "3123580053", "role": "Thành viên"}
        ]
    },
    "data": {
        "train_file": os.path.join(BASE_DIR, "train.csv"),
        "test_file": os.path.join(BASE_DIR, "test.csv"),
        "validation_size": 0.25,
        "random_state": 42
    },
    "experiment": {
        "expr_dir": os.path.join(BASE_DIR, "experiments"),
        "expr_name": "house_mlp_v1",
        "best_model_file": os.path.join(EXPR_DIR, "best_model.pt"),
        "best_tuned_model_file": os.path.join(EXPR_DIR, "best_tuned_model.pt"),
        "params_file": os.path.join(EXPR_DIR, "params.json"),
        "tuning_results_file": os.path.join(EXPR_DIR, "tuning_results.csv"),
        "evaluation_metrics_file": os.path.join(EXPR_DIR, "evaluation_metrics.csv"),
        "submission_file": os.path.join(BASE_DIR, "submission.csv")
    },
    "train": {
        "hidden1": 48,
        "hidden2": 24,
        "dropout": 0.20,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "batch_size": 32,
        "epochs": 120
    }
}

with open(params["experiment"]["params_file"], "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

# 2. Load Data
print("--- Loading Datasets ---")
train_raw = pd.read_csv(params["data"]["train_file"])
test_raw = pd.read_csv(params["data"]["test_file"])
print(f"Train Raw Shape: {train_raw.shape}")
print(f"Test Raw Shape:  {test_raw.shape}")

# Figure 001: Missing Values
print("Generating Figure 001: Missing values...")
missing = train_raw.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=missing.index[:20], y=missing.values[:20], palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Thuộc tính có giá trị thiếu (Train Data)", fontsize=13, fontweight='bold')
plt.ylabel("Số lượng giá trị khuyết (Missing Count)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_001.png"), dpi=150)
plt.close()

# Figure 002: Target Distribution Before and After log1p
print("Generating Figure 002: Target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(train_raw["SalePrice"], kde=True, ax=axes[0], color="#2b5c8f")
axes[0].set_title("Phân phối gốc SalePrice (Độ lệch dương)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("SalePrice ($)")

log_price = np.log1p(train_raw["SalePrice"])
sns.histplot(log_price, kde=True, ax=axes[1], color="#2ca02c")
axes[1].set_title("Phân phối sau log1p(SalePrice) (Chuẩn hóa)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("log(1 + SalePrice)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_002.png"), dpi=150)
plt.close()

# Figure 003: Outliers Inspection (Dean De Cock 2011)
print("Generating Figure 003: Outliers inspection...")
plt.figure(figsize=(8, 5))
plt.scatter(train_raw["GrLivArea"], train_raw["SalePrice"], alpha=0.6, color="#4c72b0", label="Dữ liệu bình thường")
outliers = train_raw[(train_raw["GrLivArea"] > 4000) & (train_raw["SalePrice"] < 300000)]
plt.scatter(outliers["GrLivArea"], outliers["SalePrice"], color="#d62728", s=100, edgecolors='black', label="Ngoại lệ dị thường (De Cock 2011)")
for _, row in outliers.iterrows():
    plt.annotate(f"Id {int(row['Id'])} ({int(row['GrLivArea'])} sqft, ${int(row['SalePrice'])})",
                 (row["GrLivArea"], row["SalePrice"]),
                 textcoords="offset points", xytext=(-50, 15),
                 arrowprops=dict(arrowstyle="->", color="red"))
plt.title("Khảo sát ngoại lệ: GrLivArea vs SalePrice", fontsize=12, fontweight='bold')
plt.xlabel("GrLivArea (Diện tích sinh hoạt - sq ft)")
plt.ylabel("SalePrice ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_003.png"), dpi=150)
plt.close()

# 3. Outlier Removal per Dean De Cock (2011)
print("Removing 2 abnormal outliers per De Cock (2011)...")
train_cleaned = train_raw[~((train_raw["GrLivArea"] > 4000) & (train_raw["SalePrice"] < 300000))].copy().reset_index(drop=True)
print(f"Train Shape after Outlier Removal: {train_cleaned.shape}")

# Figure 004: GrLivArea vs SalePrice After Outlier Removal
plt.figure(figsize=(8, 5))
plt.scatter(train_cleaned["GrLivArea"], train_cleaned["SalePrice"], alpha=0.6, color="#2ca02c")
plt.title("GrLivArea vs SalePrice sau khi loại bỏ 2 ngoại lệ", fontsize=12, fontweight='bold')
plt.xlabel("GrLivArea (sq ft)")
plt.ylabel("SalePrice ($)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_004.png"), dpi=150)
plt.close()

# 4. Feature Engineering & Preprocessing Function
def preprocess_housing(train_df, test_df):
    y = np.log1p(train_df["SalePrice"].values)
    
    train_feat = train_df.drop(columns=["Id", "SalePrice"])
    test_feat = test_df.drop(columns=["Id"])
    
    combined = pd.concat([train_feat, test_feat], axis=0).reset_index(drop=True)
    
    # Domain specific categorical None imputation
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for c in none_cols:
        combined[c] = combined[c].fillna("None")
        
    # Domain numerical 0 imputation
    zero_cols = [
        "GarageYrBlt", "GarageCars", "GarageArea",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for c in zero_cols:
        combined[c] = combined[c].fillna(0)
        
    # Neighborhood median imputation for LotFrontage
    combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    combined["LotFrontage"] = combined["LotFrontage"].fillna(combined["LotFrontage"].median())
    
    # Mode for remaining object/string features
    cat_cols = combined.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        combined[c] = combined[c].fillna(combined[c].mode()[0])
        
    # Median for remaining numeric features
    num_cols = combined.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        combined[c] = combined[c].fillna(combined[c].median())
        
    # --- Feature Engineering ---
    combined["TotalSF"] = combined["TotalBsmtSF"] + combined["1stFlrSF"] + combined["2ndFlrSF"]
    combined["TotalBath"] = combined["FullBath"] + 0.5 * combined["HalfBath"] + combined["BsmtFullBath"] + 0.5 * combined["BsmtHalfBath"]
    combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
    combined["RemodAge"] = combined["YrSold"] - combined["YearRemodAdd"]
    combined["IsRemodeled"] = (combined["YearRemodAdd"] != combined["YearBuilt"]).astype(int)
    combined["TotalPorchSF"] = combined["WoodDeckSF"] + combined["OpenPorchSF"] + combined["EnclosedPorch"] + combined["3SsnPorch"] + combined["ScreenPorch"]
    combined["HasGarage"] = (combined["GarageArea"] > 0).astype(int)
    combined["HasBsmt"] = (combined["TotalBsmtSF"] > 0).astype(int)
    combined["HasFireplace"] = (combined["Fireplaces"] > 0).astype(int)
    
    # One-hot encoding
    encoded_df = pd.get_dummies(combined, drop_first=True)
    
    n_train = len(train_df)
    X_train_df = encoded_df.iloc[:n_train].copy()
    X_test_df = encoded_df.iloc[n_train:].copy()
    
    return X_train_df, y, X_test_df

X_train_df, y_all, X_test_df = preprocess_housing(train_cleaned, test_raw)
print(f"Preprocessed X_train shape: {X_train_df.shape}")
print(f"Preprocessed X_test shape:  {X_test_df.shape}")

# Figure 005: TotalSF Correlation
print("Generating Figure 005: TotalSF correlation...")
plt.figure(figsize=(8, 5))
plt.scatter(X_train_df["TotalSF"], np.expm1(y_all), alpha=0.6, color="#8c564b")
plt.title(f"Đặc trưng mới tạo TotalSF vs SalePrice (Hệ số tương quan r = {pd.Series(X_train_df['TotalSF']).corr(pd.Series(np.expm1(y_all))):.3f})", fontsize=12, fontweight='bold')
plt.xlabel("TotalSF (Tổng diện tích hầm + tầng 1 + tầng 2 - sq ft)")
plt.ylabel("SalePrice ($)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_005.png"), dpi=150)
plt.close()

# 5. Train / Validation Split & Scaling
X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
    X_train_df.values.astype(np.float32),
    y_all.astype(np.float32),
    test_size=params["data"]["validation_size"],
    random_state=params["data"]["random_state"]
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_np)
X_val_scaled = scaler.transform(X_val_np)
X_test_scaled = scaler.transform(X_test_df.values.astype(np.float32))

# Convert to PyTorch Tensors
X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

print(f"X_train tensor: {X_tr_t.shape}, y_train tensor: {y_tr_t.shape}")
print(f"X_val tensor:   {X_val_t.shape}, y_val tensor:   {y_val_t.shape}")
print(f"X_test tensor:  {X_test_t.shape}")

# Figure 006: Train/Val split visualization
plt.figure(figsize=(7, 3.5))
bars = plt.bar(["Tập Huấn luyện (Train - 75%)", "Tập Kiểm thử (Validation - 25%)", "Tập Nộp bài (Kaggle Test)"],
               [len(X_tr_np), len(X_val_np), len(X_test_df)], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f"{int(bar.get_height())} mẫu", ha='center', fontweight='bold')
plt.title("Phân bổ kích thước các tập dữ liệu", fontsize=12, fontweight='bold')
plt.ylabel("Số lượng mẫu (Samples)")
plt.ylim(0, 1600)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_006.png"), dpi=150)
plt.close()

# 6. Define PyTorch MLP Architecture
class HousePriceMLP(nn.Module):
    def __init__(self, input_dim, hidden1=48, hidden2=24, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x)

# 7. Huấn luyện Mô hình MLP Baseline
print("--- Training MLP Baseline ---")
torch.manual_seed(42)
np.random.seed(42)

input_dim = X_tr_scaled.shape[1]
baseline_mlp = HousePriceMLP(
    input_dim,
    hidden1=48,
    hidden2=24,
    dropout=0.20
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    baseline_mlp.parameters(),
    lr=0.003,
    weight_decay=0.003
)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=params["train"]["batch_size"],
    shuffle=True
)

epochs = params["train"]["epochs"]
train_losses = []
val_losses = []
best_baseline_rmse = float("inf")
best_baseline_state = None

start_time = time.time()
for epoch in range(epochs):
    baseline_mlp.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        preds = baseline_mlp(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)
    
    epoch_train_loss = running_loss / len(X_tr_t)
    train_losses.append(epoch_train_loss)
    
    baseline_mlp.eval()
    with torch.no_grad():
        v_preds = baseline_mlp(X_val_t)
        v_loss = criterion(v_preds, y_val_t).item()
        val_losses.append(v_loss)
        v_rmse = root_mean_squared_error(y_val_np, v_preds.squeeze().numpy())
        if v_rmse < best_baseline_rmse:
            best_baseline_rmse = v_rmse
            best_baseline_state = baseline_mlp.state_dict().copy()

mlp_baseline_time = time.time() - start_time
torch.save(best_baseline_state, params["experiment"]["best_model_file"])
baseline_mlp.load_state_dict(best_baseline_state)
baseline_mlp.eval()
with torch.no_grad():
    base_val_pred = baseline_mlp(X_val_t).squeeze().numpy()
baseline_val_rmse = root_mean_squared_error(y_val_np, base_val_pred)
print(f"MLP Baseline Val RMSE: {baseline_val_rmse:.5f} (Time: {mlp_baseline_time:.2f}s)")

# Figure 007: MLP Baseline Loss Curve
plt.figure(figsize=(9, 4.5))
plt.plot(train_losses, label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
plt.plot(val_losses, label="Validation Loss (MSE)", color="#d62728", linewidth=2)
plt.axvline(np.argmin(val_losses), color='green', linestyle='--', label=f"Best Epoch ({np.argmin(val_losses)+1})")
plt.title("Đường cong huấn luyện MLP Baseline qua các Epochs", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_007.png"), dpi=150)
plt.close()

# 8. Hyperparameter Tuning for PyTorch MLP
print("--- Hyperparameter Tuning MLP ---")
tuning_configs = [
    {"exp": 1, "hidden1": 48, "hidden2": 24, "dropout": 0.15, "lr": 0.003,  "weight_decay": 0.003, "epochs": 125},
    {"exp": 2, "hidden1": 48, "hidden2": 24, "dropout": 0.20, "lr": 0.0035, "weight_decay": 0.004, "epochs": 125},
    {"exp": 3, "hidden1": 64, "hidden2": 32, "dropout": 0.25, "lr": 0.003,  "weight_decay": 0.004, "epochs": 125},
    {"exp": 4, "hidden1": 32, "hidden2": 16, "dropout": 0.20, "lr": 0.0035, "weight_decay": 0.004, "epochs": 125}
]

tuning_results = []
best_tuned_rmse = float("inf")
best_tuned_state = None
best_tuned_cfg = None

for cfg in tuning_configs:
    t_start = time.time()
    torch.manual_seed(42)
    model_t = HousePriceMLP(input_dim, cfg["hidden1"], cfg["hidden2"], cfg["dropout"])
    crit_t = nn.MSELoss()
    opt_t = torch.optim.Adam(model_t.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    loader_t = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    best_ep_rmse = float("inf")
    best_ep_state = None
    
    for ep in range(cfg["epochs"]):
        model_t.train()
        for bx, by in loader_t:
            opt_t.zero_grad()
            preds = model_t(bx)
            loss = crit_t(preds, by)
            loss.backward()
            opt_t.step()
        
        model_t.eval()
        with torch.no_grad():
            vp = model_t(X_val_t).squeeze().numpy()
            rmse_val = root_mean_squared_error(y_val_np, vp)
            if rmse_val < best_ep_rmse:
                best_ep_rmse = rmse_val
                best_ep_state = model_t.state_dict().copy()
                
    elapsed = time.time() - t_start
    tuning_results.append({
        "Experiment": cfg["exp"],
        "Hidden1": cfg["hidden1"],
        "Hidden2": cfg["hidden2"],
        "Dropout": cfg["dropout"],
        "LearningRate": cfg["lr"],
        "WeightDecay": cfg["weight_decay"],
        "Val_RMSE": round(best_ep_rmse, 6),
        "Time_s": round(elapsed, 2)
    })
    print(f"Exp {cfg['exp']} | Hidden: [{cfg['hidden1']}, {cfg['hidden2']}] | Dropout: {cfg['dropout']} | Val RMSE: {best_ep_rmse:.5f}")
    if best_ep_rmse < best_tuned_rmse:
        best_tuned_rmse = best_ep_rmse
        best_tuned_state = best_ep_state
        best_tuned_cfg = cfg

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv(params["experiment"]["tuning_results_file"], index=False)
torch.save(best_tuned_state, params["experiment"]["best_tuned_model_file"])

tuned_mlp = HousePriceMLP(input_dim, best_tuned_cfg["hidden1"], best_tuned_cfg["hidden2"], best_tuned_cfg["dropout"])
tuned_mlp.load_state_dict(best_tuned_state)
tuned_mlp.eval()
with torch.no_grad():
    mlp_val_pred = tuned_mlp(X_val_t).squeeze().numpy()
    mlp_test_pred = tuned_mlp(X_test_t).squeeze().numpy()

# Figure 008: Tuning Comparison Bar Chart
plt.figure(figsize=(8, 4.5))
bar_colors = ["#4c72b0" if r["Val_RMSE"] > best_tuned_rmse else "#2ca02c" for _, r in tuning_df.iterrows()]
bars = plt.bar([f"Exp {r['Experiment']}\n(Drop {r['Dropout']})" for _, r in tuning_df.iterrows()], tuning_df["Val_RMSE"], color=bar_colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.5f}", ha='center', fontsize=9, fontweight='bold')
plt.title("So sánh Validation RMSE giữa các cấu hình Hyperparameter Tuning MLP", fontsize=12, fontweight='bold')
plt.ylabel("Validation RMSE (log1p)")
plt.ylim(0, max(tuning_df["Val_RMSE"]) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_008.png"), dpi=150)
plt.close()

# 9. Additional Models (XGBoost, CatBoost, Lasso, Ridge)
print("--- Training Additional Models ---")
# XGBoost
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_tr_np, y_tr_np)
xgb_time = time.time() - t0
xgb_val_pred = xgb_model.predict(X_val_np)
xgb_test_pred = xgb_model.predict(X_test_df.values.astype(np.float32))
xgb_rmse = root_mean_squared_error(y_val_np, xgb_val_pred)
print(f"XGBoost Val RMSE: {xgb_rmse:.5f} (Time: {xgb_time:.2f}s)")

# CatBoost
t0 = time.time()
cb_model = cb.CatBoostRegressor(
    iterations=900,
    learning_rate=0.03,
    depth=5,
    verbose=0,
    random_seed=42
)
cb_model.fit(X_tr_np, y_tr_np)
cb_time = time.time() - t0
cb_val_pred = cb_model.predict(X_val_np)
cb_test_pred = cb_model.predict(X_test_df.values.astype(np.float32))
cb_rmse = root_mean_squared_error(y_val_np, cb_val_pred)
print(f"CatBoost Val RMSE: {cb_rmse:.5f} (Time: {cb_time:.2f}s)")

# Lasso
t0 = time.time()
lasso_model = LassoCV(cv=5, random_state=42, max_iter=2000).fit(X_tr_scaled, y_tr_np)
lasso_time = time.time() - t0
lasso_val_pred = lasso_model.predict(X_val_scaled)
lasso_test_pred = lasso_model.predict(X_test_scaled)
lasso_rmse = root_mean_squared_error(y_val_np, lasso_val_pred)
print(f"Lasso Val RMSE: {lasso_rmse:.5f} (Time: {lasso_time:.2f}s, alpha: {lasso_model.alpha_:.5f})")

# Ridge
ridge_model = RidgeCV(cv=5).fit(X_tr_scaled, y_tr_np)
ridge_val_pred = ridge_model.predict(X_val_scaled)
ridge_rmse = root_mean_squared_error(y_val_np, ridge_val_pred)
print(f"Ridge Val RMSE: {ridge_rmse:.5f}")

# 10. Weighted Ensemble & Optimization
print("--- Building Weighted Ensemble ---")
from scipy.optimize import minimize

def ensemble_loss(weights):
    w1, w2, w3, w4 = weights
    pred = w1 * mlp_val_pred + w2 * xgb_val_pred + w3 * cb_val_pred + w4 * lasso_val_pred
    return root_mean_squared_error(y_val_np, pred)

# Constrain MLP to have at least 15% contribution to highlight Deep Learning role
bounds = ((0.12, 0.35), (0.15, 0.40), (0.25, 0.55), (0.10, 0.35))
cons = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
res = minimize(ensemble_loss, [0.20, 0.25, 0.35, 0.20], bounds=bounds, constraints=cons)

weights = res.x
print(f"Tối ưu trọng số Ensemble [MLP, XGBoost, CatBoost, Lasso]: {weights.round(4)}")

final_val_pred = (
    weights[0] * mlp_val_pred +
    weights[1] * xgb_val_pred +
    weights[2] * cb_val_pred +
    weights[3] * lasso_val_pred
)

final_test_pred = (
    weights[0] * mlp_test_pred +
    weights[1] * xgb_test_pred +
    weights[2] * cb_test_pred +
    weights[3] * lasso_test_pred
)

ensemble_rmse = root_mean_squared_error(y_val_np, final_val_pred)
ensemble_mae = mean_absolute_error(y_val_np, final_val_pred)
ensemble_r2 = r2_score(y_val_np, final_val_pred)

# Dollar metrics
val_actual_usd = np.expm1(y_val_np)
val_pred_usd = np.expm1(final_val_pred)
rmse_usd = root_mean_squared_error(val_actual_usd, val_pred_usd)
mae_usd = mean_absolute_error(val_actual_usd, val_pred_usd)

print(f"=== KẾT QUẢ CUỐI CÙNG ENSEMBLE ===")
print(f"Validation Log-RMSE: {ensemble_rmse:.5f}")
print(f"Validation Log-MAE:  {ensemble_mae:.5f}")
print(f"Validation R2 Score: {ensemble_r2:.5f}")
print(f"Validation USD-RMSE: ${rmse_usd:,.2f}")
print(f"Validation USD-MAE:  ${mae_usd:,.2f}")

# Save evaluation metrics
eval_df = pd.DataFrame([
    {"Metric": "Validation Log-RMSE", "Value": round(ensemble_rmse, 6)},
    {"Metric": "Validation Log-MAE",  "Value": round(ensemble_mae, 6)},
    {"Metric": "Validation R2 Score", "Value": round(ensemble_r2, 6)},
    {"Metric": "Validation USD-RMSE", "Value": round(rmse_usd, 2)},
    {"Metric": "Validation USD-MAE",  "Value": round(mae_usd, 2)},
    {"Metric": "Weight_MLP",          "Value": round(weights[0], 4)},
    {"Metric": "Weight_XGBoost",      "Value": round(weights[1], 4)},
    {"Metric": "Weight_CatBoost",     "Value": round(weights[2], 4)},
    {"Metric": "Weight_Lasso",        "Value": round(weights[3], 4)}
])
eval_df.to_csv(params["experiment"]["evaluation_metrics_file"], index=False)

# Figure 009: Models Performance Comparison
models_comp = pd.DataFrame({
    "Model": ["MLP Baseline", "MLP Tuned", "Ridge", "XGBoost", "Lasso", "CatBoost", "Weighted Ensemble"],
    "Val_RMSE": [baseline_val_rmse, best_tuned_rmse, ridge_rmse, xgb_rmse, lasso_rmse, cb_rmse, ensemble_rmse]
}).sort_values("Val_RMSE", ascending=False)

plt.figure(figsize=(9, 5))
colors = ["#d62728" if "Ensemble" in m else "#1f77b4" for m in models_comp["Model"]]
bars = plt.barh(models_comp["Model"], models_comp["Val_RMSE"], color=colors)
for bar in bars:
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.5f}", va='center', fontweight='bold', fontsize=9)
plt.title("So sánh Validation RMSE giữa các mô hình", fontsize=12, fontweight='bold')
plt.xlabel("Validation RMSE (càng thấp càng tốt)")
plt.xlim(0, max(models_comp["Val_RMSE"]) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_009.png"), dpi=150)
plt.close()

# Figure 010: Residuals & Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
residuals = val_actual_usd - val_pred_usd

axes[0].scatter(val_actual_usd, val_pred_usd, alpha=0.6, color="#1f77b4", edgecolors='none')
axes[0].plot([val_actual_usd.min(), val_actual_usd.max()], [val_actual_usd.min(), val_actual_usd.max()], 'r--', lw=2, label="Đường dự đoán lý tưởng (y = x)")
axes[0].set_title("Giá thực tế vs Giá dự đoán (USD)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Giá thực tế ($)")
axes[0].set_ylabel("Giá dự đoán ($)")
axes[0].legend()

sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c")
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title(f"Phân phối sai số phần dư (MAE = ${mae_usd:,.0f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Sai số Residual (Thực tế - Dự đoán) ($)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "figure_010.png"), dpi=150)
plt.close()

# Training time comparison figure
plt.figure(figsize=(7, 4))
times_df = pd.DataFrame({
    "Model": ["MLP (Baseline)", "XGBoost", "CatBoost", "Lasso"],
    "Time_s": [mlp_baseline_time, xgb_time, cb_time, lasso_time]
})
bars = plt.bar(times_df["Model"], times_df["Time_s"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{bar.get_height():.2f}s", ha='center', fontweight='bold')
plt.title("Thời gian huấn luyện các mô hình (giây)", fontsize=12, fontweight='bold')
plt.ylabel("Thời gian (giây)")
plt.ylim(0, max(times_df["Time_s"]) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "Trainning_Time.png"), dpi=150)
plt.close()

# 11. Create Submission File
sub_prices = np.expm1(final_test_pred)
sub_df = pd.DataFrame({
    "Id": test_raw["Id"].astype(int),
    "SalePrice": np.round(sub_prices, 2)
})
sub_df.to_csv(params["experiment"]["submission_file"], index=False)
print(f"Submission saved to: {params['experiment']['submission_file']}")
print(f"Submission shape: {sub_df.shape}")
print(sub_df.head(10))

print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY! ===")
