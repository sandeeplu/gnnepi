# =========================
# 3D Molecular Graph SchNet Training
# Using CSV with atom coordinates + masks + SMILES
# =========================

import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib.offsetbox import AnchoredText

# --- Setup and Reproducibility ---
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_set(42)
torch.use_deterministic_algorithms(True)
generator = torch.Generator().manual_seed(42)

# --- Data Loading and Log Transform ---
epsilon = 1e-6
df = pd.read_csv('epi_133725_3D_exact.csv')
df["log_electrophilicity_index"] = np.log(df["electrophilicity_index"] + epsilon)

# --- Graph Construction from atomic positions and SMILES ---
graph_list = []
failed_mols = []

for idx, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            failed_mols.append(row['smiles'])
            continue
        mol = Chem.AddHs(mol)
        atoms = mol.GetAtoms()
        n_atoms = int(row['n_atoms'])
        z = []
        pos = []
        for j in range(1, n_atoms + 1):
            x = row[f"atom{j}_x"]
            y = row[f"atom{j}_y"]
            z_pos = row[f"atom{j}_z"]
            pos.append([x, y, z_pos])
            if j-1 < len(atoms):
                z.append(atoms[j-1].GetAtomicNum())
            else:
                z.append(6)  # fallback atomic number (carbon)
        if len(z) == 0:
            continue
        z = torch.tensor(z, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y_val = row['log_electrophilicity_index']
        if torch.isnan(torch.tensor([y_val])) or torch.isinf(torch.tensor([y_val])):
            continue
        y = torch.tensor([y_val], dtype=torch.float)

        # Print check: number of atoms and coordinate dimensions
        print(f"Mol idx {idx}: n_atoms={n_atoms}, positions count={len(pos)}, total coords={3*len(pos)}")

        graph_list.append(Data(z=z, pos=pos, y=y))
    except Exception as e:
        failed_mols.append(row['smiles'])
        continue

print(f"Successfully embedded {len(graph_list)} molecules.")
print(f"Failed to embed {len(failed_mols)} molecules.")

# --- Dataset Splitting ---
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Model and Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SchNet(
    hidden_channels=64,
    num_filters=256,
    num_interactions=5,
    num_gaussians=50,
    cutoff=6,  # typical default
    readout='add'
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.000713, weight_decay=1.569e-06)

# --- Training and Testing Functions ---
def train(loader):
    model.train()
    total_loss = total_mae = total_samples = 0
    for data in loader:
        data = data.to(device)
        if torch.isnan(data.y).any() or torch.isinf(data.y).any():
            continue
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        if torch.isnan(out).any() or torch.isinf(out).any():
            continue
        mse_loss = F.mse_loss(out, data.y.view(-1,1))
        mae_loss = F.l1_loss(out, data.y.view(-1,1))
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += mse_loss.item() * data.num_graphs
        total_mae += mae_loss.item() * data.num_graphs
        total_samples += data.num_graphs
    if total_samples == 0:
        return float('nan'), float('nan'), float('nan')
    return np.sqrt(total_loss/total_samples), total_mae/total_samples, total_loss/total_samples

@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse), mean_absolute_error(actuals, preds), mse

# --- Training Loop ---
os.makedirs('models', exist_ok=True)
best_rmse = float('inf')
epochs = 500
patience = 20
counter = 0

train_rmse, train_mae, train_mse = [], [], []
test_rmse, test_mae, test_mse = [], [], []

for epoch in range(epochs):
    tr_rmse, tr_mae, tr_mse = train(train_loader)
    te_rmse, te_mae, te_mse = test(test_loader)
    train_rmse.append(tr_rmse)
    train_mae.append(tr_mae)
    train_mse.append(tr_mse)
    test_rmse.append(te_rmse)
    test_mae.append(te_mae)
    test_mse.append(te_mse)
    print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {tr_rmse:.4f}, Test RMSE: {te_rmse:.4f}")
    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --- Load Best Model ---
if os.path.exists("models/best_model.pth"):
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

# --- Evaluation ---
@torch.no_grad()
def evaluate(loader):
    preds_log, actuals_log = [], []
    preds_orig, actuals_orig = [], []
    for data in loader:
        data = data.to(device)
        out_log = model(data.z, data.pos, data.batch).view(-1)
        y_log = data.y.view(-1)
        preds_log.append(out_log.cpu().numpy())
        actuals_log.append(y_log.cpu().numpy())
        preds_orig.append(np.exp(out_log.cpu().numpy()))
        actuals_orig.append(np.exp(y_log.cpu().numpy()))
    preds_log = np.concatenate(preds_log)
    actuals_log = np.concatenate(actuals_log)
    preds_orig = np.concatenate(preds_orig)
    actuals_orig = np.concatenate(actuals_orig)
    df_eval = pd.DataFrame({
        'pred_log': preds_log,
        'actual_log': actuals_log,
        'pred_orig': preds_orig,
        'actual_orig': actuals_orig
    })
    r2_log = r2_score(actuals_log, preds_log)
    mse_log = mean_squared_error(actuals_log, preds_log)
    mae_log = mean_absolute_error(actuals_log, preds_log)
    r2_orig = r2_score(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)
    return df_eval, (r2_log, mse_log, mae_log), (r2_orig, mse_orig, mae_orig)

train_results, (train_r2_log, train_mse_log, train_mae_log), (train_r2_orig, train_mse_orig, train_mae_orig) = evaluate(train_loader)
test_results, (test_r2_log, test_mse_log, test_mae_log), (test_r2_orig, test_mse_orig, test_mae_orig) = evaluate(test_loader)

test_results.to_csv("predictions_log_and_original.csv", index=False)

print("\n\U0001F4D8 Final Performance Metrics")
print("-" * 50)
print("Log Scale Evaluation (model output space)")
print(f"Train MAE   : {train_mae_log:.4f}")
print(f"Test  MAE   : {test_mae_log:.4f}")
print(f"Train MSE   : {train_mse_log:.4f}")
print(f"Test  MSE   : {test_mse_log:.4f}")
print(f"Train RMSE  : {np.sqrt(train_mse_log):.4f}")
print(f"Test  RMSE  : {np.sqrt(test_mse_log):.4f}")
print(f"Train R²    : {train_r2_log:.4f}")
print(f"Test  R²    : {test_r2_log:.4f}")
print("-" * 50)
print("Original Scale Evaluation (after exp transform)")
print(f"Train MAE   : {train_mae_orig:.4f}")
print(f"Test  MAE   : {test_mae_orig:.4f}")
print(f"Train MSE   : {train_mse_orig:.4f}")
print(f"Test  MSE   : {test_mse_orig:.4f}")
print(f"Train RMSE  : {np.sqrt(train_mse_orig):.4f}")
print(f"Test  RMSE  : {np.sqrt(test_mse_orig):.4f}")
print(f"Train R²    : {train_r2_orig:.4f}")
print(f"Test  R²    : {test_r2_orig:.4f}")
print("-" * 50)

# =========================
# PLOTTING SECTION
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2
})

os.makedirs('plots', exist_ok=True)

# --- RMSE over epochs ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', color='red')
plt.plot(range(1, len(test_rmse)+1), test_rmse, label='Test RMSE', color='blue')
plt.xlabel('Epoch'); plt.ylabel('RMSE')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_rmse.png'); plt.close()

# ---------- LOG SCALE: Actual vs Predicted & Residuals -----------
plt.figure(figsize=(14, 6), dpi=300)
# -- Actual vs Predicted (log)
plt.subplot(1, 2, 1)
plt.scatter(
    test_results['actual_log'], test_results['pred_log'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.regplot(
    data=train_results, x='actual_log', y='pred_log', color='red',
    scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'}
)
plt.xlabel('Actual (log)', fontsize=20, fontweight='bold')
plt.ylabel('Predicted (log)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=20)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_log:.4f}\nR² (Test): {test_r2_log:.4f}",
    loc='upper left', prop=dict(size=18, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

# -- Residuals (log)
plt.subplot(1, 2, 2)
plt.scatter(
    test_results['actual_log'], test_results['actual_log'] - test_results['pred_log'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.scatterplot(
    x=train_results['actual_log'],
    y=train_results['actual_log'] - train_results['pred_log'],
    color='red', alpha=0.3, edgecolor='black', label='Train'
)
plt.xlabel('Actual (log)', fontsize=20, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(loc='upper left', frameon=False, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'SchNet',
    transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_log_hd.png', dpi=600)
plt.close()

# ---------- LOG SCALE: KDE Histogram of Errors -----------
plt.figure(figsize=(7, 6), dpi=300)
sns.histplot(
    test_results['actual_log'] - test_results['pred_log'],
    kde=True, color='blue', bins=30, edgecolor='black'
)
plt.xlabel('Prediction Error (log)', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/error_histogram_kde_log_hd.png', dpi=600)
plt.close()

# ---------- ORIGINAL SCALE: Actual vs Predicted & Residuals -----------
plt.figure(figsize=(14, 6), dpi=300)
# -- Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(
    test_results['actual_orig'], test_results['pred_orig'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.regplot(
    data=train_results, x='actual_orig', y='pred_orig', color='red',
    scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'}
)
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Predicted', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=20)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_orig:.4f}\nR² (Test): {test_r2_orig:.4f}",
    loc='upper left', prop=dict(size=18, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

# -- Residuals
plt.subplot(1, 2, 2)
plt.scatter(
    test_results['actual_orig'], test_results['actual_orig'] - test_results['pred_orig'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.scatterplot(
    x=train_results['actual_orig'],
    y=train_results['actual_orig'] - train_results['pred_orig'],
    color='red', alpha=0.3, edgecolor='black', label='Train'
)
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(loc='upper left', frameon=False, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'SchNet',
    transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_original_hd.png', dpi=600)
plt.close()

# ---------- ORIGINAL SCALE: KDE Histogram of Errors -----------
plt.figure(figsize=(7, 6), dpi=300)
sns.histplot(
    test_results['actual_orig'] - test_results['pred_orig'],
    kde=True, color='blue', bins=30, edgecolor='black'
)
plt.xlabel('Prediction Error', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/error_histogram_kde_original_hd.png', dpi=600)
plt.close()


# --- Actual vs Predicted and Residuals (Log and Original) ---
# ... (You can copy all the plotting code from your original script here)
# For brevity, all remaining plotting code is identical to your original script

print("\nTraining and evaluation finished. Plots saved in 'plots/' folder.")
