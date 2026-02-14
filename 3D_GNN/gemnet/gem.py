#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GemNet-compatible training script with cutoff=6 Å.
 - Node features explicit (atomic, chemical, geometric)
 - Edge features filtered by cutoff
 - Full Dataloader, model, evaluation, and plotting
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import itertools

#Time
start_time = time.time()

# === Seed & Reproducibility ===
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

# === Parameters ===
MAX_ATOMS = 29
CUTOFF = 6.0  # Ångström cutoff for edges
target_col = "log_electrophilicity_index"
epsilon = 1e-6

# Load CSV
df = pd.read_csv("../epi_133725_3D_exact.csv")
if target_col not in df.columns and "electrophilicity_index" in df.columns:
    df[target_col] = np.log(df["electrophilicity_index"] + epsilon)

# Atom features
def atom_features(atom):
    Z = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybrid = int(atom.GetHybridization())
    is_aromatic = int(atom.GetIsAromatic())
    return [Z, degree, formal_charge, hybrid, is_aromatic]

# Build GemNet-style graphs
graph_list, failed = [], []

for idx, row in df.iterrows():
    try:
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append((smiles, "invalid SMILES"))
            continue
        mol = Chem.AddHs(mol)
        atoms = list(mol.GetAtoms())

        n_atoms = int(row["n_atoms"])
        if n_atoms == 0 or n_atoms > MAX_ATOMS:
            failed.append((smiles, "invalid atom count"))
            continue

        pos_list, z_list, x_feats = [], [], []
        for j in range(1, n_atoms + 1):
            x_key, y_key, z_key = f"atom{j}_x", f"atom{j}_y", f"atom{j}_z"
            if pd.isna(row[x_key]) or pd.isna(row[y_key]) or pd.isna(row[z_key]):
                continue
            pos_list.append([float(row[x_key]), float(row[y_key]), float(row[z_key])])
            if j-1 < len(atoms):
                a = atoms[j-1]
                z_list.append(a.GetAtomicNum())
                x_feats.append(atom_features(a))
            else:
                # fallback if atom info missing
                z_list.append(6)  # carbon
                x_feats.append([6, 0, 0, 0, 0])

        if not pos_list or not z_list or not x_feats:
            failed.append((smiles, "missing atom/feature/position data"))
            continue

        pos = torch.tensor(pos_list, dtype=torch.float)
        z = torch.tensor(z_list, dtype=torch.long)
        x = torch.tensor(x_feats, dtype=torch.float)

        # Edges: all atom pairs within cutoff
        edges, edge_dists = [], []
        for i, j in itertools.combinations(range(len(pos_list)), 2):
            dist = torch.norm(pos[i] - pos[j]).item()
            if dist <= CUTOFF:
                edges.extend([[i, j], [j, i]])
                edge_dists.extend([[dist], [dist]])

        if not edges:
            failed.append((smiles, "no edges within cutoff"))
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_dists, dtype=torch.float)

        # Target
        try:
            y_val = float(row.get(target_col, np.nan))
        except:
            failed.append((smiles, "invalid target (cannot convert to float)"))
            continue

        if np.isnan(y_val) or np.isinf(y_val):
            failed.append((smiles, "invalid target"))
            continue

        y = torch.tensor([y_val], dtype=torch.float)

        # Build Data object
        data = Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph_list.append(data)

        # Optional: print check
        print(f"Mol idx {idx}: n_atoms={n_atoms}, positions count={len(pos_list)}, total coords={3*len(pos_list)}")

    except Exception as e:
        failed.append((row.get("smiles", None), f"exception: {e}"))
        continue

print(f"✅ Built {len(graph_list)} GemNet-style graphs (cutoff={CUTOFF} Å).")
print(f"❌ Skipped {len(failed)} molecules. Examples: {failed[:5]}")


###Data split
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === GemNet-inspired layers ===
class RBFLayer(nn.Module):
    """Radial basis expansion for distances."""
    def __init__(self, num_rbf=64, cutoff=6.0, gamma=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.centers = torch.linspace(0, cutoff, num_rbf)
        self.gamma = gamma

    def forward(self, distances):
        d = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * (d - self.centers.to(d.device))**2)


class GemNetBlock(MessagePassing):
    def __init__(self, node_dim, hidden_dim=128, num_rbf=64):
        super().__init__(aggr='add')
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.rbf_layer = RBFLayer(num_rbf=num_rbf)
        self.rbf_emb = nn.Linear(num_rbf, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, pos):
        row, col = edge_index
        x_emb = self.node_emb(x)

        # compute pair distances
        dist = torch.norm(pos[row] - pos[col], dim=-1)
        rbf = self.rbf_layer(dist)
        edge_attr = self.rbf_emb(rbf)

        return self.propagate(edge_index, x=x_emb, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg)

    def update(self, aggr_out):
        return self.node_update(aggr_out)


class GemNetModel(nn.Module):
    def __init__(self, node_dim, hidden_dim=128, num_rbf=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            GemNetBlock(node_dim, hidden_dim, num_rbf) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        for layer in self.layers:
            x = layer(x, edge_index, pos)
        batch = data.batch
        x = scatter(x, batch, dim=0, reduce='add')
        out = self.fc(x)
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(graph_list) == 0:
    raise RuntimeError("No graphs were created. Check dataset and preprocessing.")
if getattr(graph_list[0], 'x', None) is None:
    raise RuntimeError("Node features 'x' missing from first graph. Ensure x_feats are created.")

node_dim = graph_list[0].x.shape[1]

# ✅ initialize the GemNet-like model (edge_dim not needed here)
model = GemNetModel(node_dim=node_dim, hidden_dim=256, num_rbf=64, num_layers=4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00011953092590523518, weight_decay=1.8351393227040611e-06)

def train(loader):
    model.train()
    loss_sum = mae_sum = samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        mse_loss = F.mse_loss(out, data.y.view(-1))
        mae_loss = F.l1_loss(out, data.y.view(-1))
        mse_loss.backward()
        optimizer.step()
        n_graphs = data.num_graphs
        loss_sum += mse_loss.item() * n_graphs
        mae_sum += mae_loss.item() * n_graphs
        samples += n_graphs
    return np.sqrt(loss_sum / samples), mae_sum / samples, loss_sum / samples

@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    return np.sqrt(mse), mae, mse, r2

@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds_log, actuals_log = [], []
    preds_orig, actuals_orig = [], []
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1)
        y = data.y.view(-1)
        preds_log.append(out.cpu().numpy())
        actuals_log.append(y.cpu().numpy())
        preds_orig.append(np.exp(out.cpu().numpy()))
        actuals_orig.append(np.exp(y.cpu().numpy()))
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

os.makedirs('models', exist_ok=True)
best_rmse = float('inf'); patience = 20; counter = 0
train_rmse, train_mae, train_mse, train_r2 = [], [], [], []
test_rmse, test_mae, test_mse, test_r2 = [], [], [], []
for epoch in range(1, 301):
    tr_rmse, tr_mae, tr_mse, tr_r2 = test(train_loader)
    te_rmse, te_mae, te_mse, te_r2 = test(test_loader)
    train_rmse.append(tr_rmse); train_mae.append(tr_mae); train_mse.append(tr_mse); train_r2.append(tr_r2)
    test_rmse.append(te_rmse); test_mae.append(te_mae); test_mse.append(te_mse); test_r2.append(te_r2)
    print(f"Epoch {epoch:4d} | Train RMSE: {tr_rmse:.4f} | Test RMSE: {te_rmse:.4f}")
    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), "models/best_model_gemnet.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break
    train(train_loader)

if os.path.exists("models/best_model_gemnet.pth"):
    state = torch.load("models/best_model_gemnet.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

train_results, (train_r2_log, train_mse_log, train_mae_log), (train_r2_orig, train_mse_orig, train_mae_orig) = evaluate(train_loader)
test_results, (test_r2_log, test_mse_log, test_mae_log), (test_r2_orig, test_mse_orig, test_mae_orig) = evaluate(test_loader)
test_results.to_csv("predictions_gemnet.csv", index=False)
print(f"\nGemNet Test Log RMSE: {np.sqrt(test_mse_log):.4f}, MAE: {test_mae_log:.4f}, R2: {test_r2_log:.4f}, Test Original RMSE: {np.sqrt(test_mse_orig):.4f}")

print("\n Final Performance Metrics")
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



# === Publication-quality Plotting ===
mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 16, 'figure.dpi': 300,
    'savefig.dpi': 600, 'axes.linewidth': 1.5, 'lines.linewidth': 2
})
os.makedirs('plots', exist_ok=True)
#Epoch vs RMSE
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', color='red')
plt.plot(range(1, len(test_rmse)+1), test_rmse, label='Test RMSE', color='blue')
plt.xlabel('Epoch'); plt.ylabel('RMSE')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_rmse_egnn.png'); plt.close()

# LOG SCALE: Actual vs Predicted & Residuals
plt.figure(figsize=(14, 6), dpi=300)
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
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=18)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_log:.4f}\nR² (Test): {test_r2_log:.4f}",
    loc='upper left', prop=dict(size=16, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

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
plt.legend(loc='upper left', frameon=False, fontsize=18)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'GemNet', transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_log_hd_egnn.png', dpi=300)
plt.close()

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
plt.savefig('plots/error_histogram_kde_log_hd_egnn.png', dpi=300)
plt.close()

# ORIGINAL SCALE: Actual vs Predicted & Residuals
plt.figure(figsize=(14, 6), dpi=300)
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
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=18)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_orig:.4f}\nR² (Test): {test_r2_orig:.4f}",
    loc='upper left', prop=dict(size=16, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

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
plt.legend(loc='upper left', frameon=False, fontsize=18)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'GemNet', transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_original_hd_egnn.png', dpi=300)
plt.close()

plt.figure(figsize=(7, 6), dpi=300)
errors = test_results['actual_orig'] - test_results['pred_orig']
xmin, xmax = errors.min(), errors.max()
xticks = np.linspace(xmin, xmax, num=7)
sns.histplot(errors, kde=True, color='blue', bins=30, edgecolor='black')
plt.xlabel('Prediction Error', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')
plt.xticks(xticks)
plt.xlim(xmin, xmax)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/error_histogram_kde_original_hd_egnn.png', dpi=300)
plt.close()


print(f"Execution time: {(time.time() - start_time)/60:.2f} minutes")
