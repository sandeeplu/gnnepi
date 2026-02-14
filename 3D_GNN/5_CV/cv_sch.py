import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem

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

        graph_list.append(Data(z=z, pos=pos, y=y))
    except Exception:
        failed_mols.append(row['smiles'])
        continue

print(f"Successfully embedded {len(graph_list)} molecules.")
print(f"Failed to embed {len(failed_mols)} molecules.")

# --- 5-Fold Cross Validation Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'rmse_log': [], 'mae_log': [], 'mse_log': [], 'r2_log': [],
    'rmse_orig': [], 'mae_orig': [], 'mse_orig': [], 'r2_orig': []
}
overall_start_time = time.time()

def train(model, loader, optimizer):
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
    return np.sqrt(total_loss / total_samples), total_mae / total_samples, total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
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
    targets_log = np.concatenate(actuals_log)
    preds_orig = np.concatenate(preds_orig)
    targets_orig = np.concatenate(actuals_orig)
    return {
        'rmse_log': np.sqrt(mean_squared_error(targets_log, preds_log)),
        'mae_log': mean_absolute_error(targets_log, preds_log),
        'mse_log': mean_squared_error(targets_log, preds_log),
        'r2_log': r2_score(targets_log, preds_log),
        'rmse_orig': np.sqrt(mean_squared_error(targets_orig, preds_orig)),
        'mae_orig': mean_absolute_error(targets_orig, preds_orig),
        'mse_orig': mean_squared_error(targets_orig, preds_orig),
        'r2_orig': r2_score(targets_orig, preds_orig)
    }

# --- Cross-Validation Loop ---
for fold, (train_idx, test_idx) in enumerate(kfold.split(graph_list)):
    print(f"\nFOLD {fold + 1} ----------------------------")
    fold_start_time = time.time()

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(graph_list, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(graph_list, batch_size=32, sampler=test_sampler)

    model = SchNet(
        hidden_channels=64,
        num_filters=256,
        num_interactions=5,
        num_gaussians=50,
        cutoff=6.0,
        readout='add'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007134394, weight_decay=1.5694399e-6)

    best_rmse = float('inf')
    patience = 20
    counter = 0
    best_state_dict = None
    epochs = 500

    for epoch in range(epochs):
        train_rmse, train_mae, train_mse = train(model, train_loader, optimizer)
        val_metrics = evaluate(model, test_loader)
        val_rmse_log = val_metrics['rmse_log']
        if np.isnan(val_rmse_log):
            print("NaN detected during validation, stopping early.")
            break
        if val_rmse_log < best_rmse:
            best_rmse = val_rmse_log
            counter = 0
            best_state_dict = model.state_dict()
        else:
            counter += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse_log:.4f}")
        if counter >= patience:
            print("Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    fold_metrics = evaluate(model, test_loader)
    for key in metrics:
        metrics[key].append(fold_metrics[key])

    print(f"Fold {fold + 1} metrics (log)  RMSE={fold_metrics['rmse_log']:.4f}, MAE={fold_metrics['mae_log']:.4f}, R2={fold_metrics['r2_log']:.4f}")
    print(f"Fold {fold + 1} metrics (orig) RMSE={fold_metrics['rmse_orig']:.4f}, MAE={fold_metrics['mae_orig']:.4f}, R2={fold_metrics['r2_orig']:.4f}")
    print(f"Fold {fold + 1} completed in {(time.time() - fold_start_time):.2f} seconds.")

# --- Aggregated Final Results ---
print("\nFinal 5-Fold Cross Validation Results (mean ± std):")
for key in ['rmse_log', 'mae_log', 'mse_log', 'r2_log']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (log scale)")
for key in ['rmse_orig', 'mae_orig', 'mse_orig', 'r2_orig']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (original scale)")
print(f"\nTotal runtime: {(time.time() - overall_start_time):.2f} seconds")
