import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn import Sequential as Seq, Linear, BatchNorm1d, ReLU

# --- Setup and Reproducibility ---
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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# --- Data Loading and Log Transform ---
epsilon = 1e-6
df = pd.read_csv('../../epi.csv')
df["log_electrophilicity_index"] = np.log(df["electrophilicity_index"] + epsilon)

# --- Atom types for one-hot encoding ---
atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
atom_type_to_idx = {anum: idx for idx, anum in enumerate(atom_types)}
num_atom_types = len(atom_types)

# --- Graph Construction for GINE ---
graph_list = []
for i, smile in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0:
        continue
    AllChem.UFFOptimizeMolecule(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [0] * num_atom_types
        anum = atom.GetAtomicNum()
        if anum in atom_type_to_idx:
            feat[atom_type_to_idx[anum]] = 1
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        edge_index += [[i1, i2], [i2, i1]]
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr += [[bond_type], [bond_type]]
    if len(edge_index) == 0:
        continue
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y_val = df['log_electrophilicity_index'].iloc[i]
    if np.isnan(y_val) or np.isinf(y_val):
        continue
    y = torch.tensor([y_val], dtype=torch.float)
    graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

print(f"Successfully embedded {len(graph_list)} molecules.")

# --- GINE Model ---
class GINENet(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_dim=128, num_targets=1, dropout=0.1):
        super().__init__()
        self.conv1 = GINEConv(
            Seq(
                Linear(node_feature_size, hidden_dim),
                BatchNorm1d(hidden_dim), ReLU(),
                Linear(hidden_dim, hidden_dim), ReLU()
            ),
            edge_dim=edge_feature_size
        )
        self.conv2 = GINEConv(
            Seq(
                Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                Linear(hidden_dim, hidden_dim), ReLU()
            ),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Seq(
                Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                Linear(hidden_dim, hidden_dim), ReLU()
            ),
            edge_dim=edge_feature_size
        )
        self.conv4 = GINEConv(
            Seq(
                Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                Linear(hidden_dim, hidden_dim), ReLU()
            ),
            edge_dim=edge_feature_size
        )
        self.lin1 = Linear(hidden_dim * 4, hidden_dim * 4)
        self.lin2 = Linear(hidden_dim * 4, num_targets)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)
        h4_pool = global_add_pool(h4, batch)
        h = torch.cat([h1_pool, h2_pool, h3_pool, h4_pool], dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        return h

# --- Training Function ---
def train(model, loader, optimizer, device):
    model.train()
    total_loss = total_mae = total_samples = 0
    for data in loader:
        data = data.to(device)
        if torch.isnan(data.y).any() or torch.isinf(data.y).any():
            continue
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        if torch.isnan(out).any() or torch.isinf(out).any():
            continue
        mse_loss = F.mse_loss(out, data.y.view(-1, 1))
        mae_loss = F.l1_loss(out, data.y.view(-1, 1))
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += mse_loss.item() * data.num_graphs
        total_mae += mae_loss.item() * data.num_graphs
        total_samples += data.num_graphs
    if total_samples == 0:
        return float('nan'), float('nan'), float('nan')
    return np.sqrt(total_loss / total_samples), total_mae / total_samples, total_loss / total_samples

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_log, actuals_log = [], []
    preds_orig, actuals_orig = [], []
    for data in loader:
        data = data.to(device)
        out_log = model(data.x, data.edge_index, data.edge_attr, data.batch).view(-1)
        y_log = data.y.view(-1)
        preds_log.append(out_log.cpu().numpy())
        actuals_log.append(y_log.cpu().numpy())
        preds_orig.append(np.exp(out_log.cpu().numpy()))
        actuals_orig.append(np.exp(y_log.cpu().numpy()))
    preds_log = np.concatenate(preds_log)
    targets_log = np.concatenate(actuals_log)
    preds_orig = np.concatenate(preds_orig)
    targets_orig = np.concatenate(actuals_orig)
    rmse_log = np.sqrt(mean_squared_error(targets_log, preds_log))
    mae_log = mean_absolute_error(targets_log, preds_log)
    mse_log = mean_squared_error(targets_log, preds_log)
    r2_log = r2_score(targets_log, preds_log)
    rmse_orig = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    mae_orig = mean_absolute_error(targets_orig, preds_orig)
    mse_orig = mean_squared_error(targets_orig, preds_orig)
    r2_orig = r2_score(targets_orig, preds_orig)
    return {
        'rmse_log': rmse_log, 'mae_log': mae_log, 'mse_log': mse_log, 'r2_log': r2_log,
        'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'mse_orig': mse_orig, 'r2_orig': r2_orig
    }

# --- 5-Fold Cross Validation ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'rmse_log': [], 'mae_log': [], 'mse_log': [], 'r2_log': [],
    'rmse_orig': [], 'mae_orig': [], 'mse_orig': [], 'r2_orig': []
}
overall_start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(kfold.split(graph_list)):
    print(f"\nFOLD {fold + 1} ----------------------------")
    fold_start_time = time.time()
    train_loader = DataLoader(graph_list, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    test_loader = DataLoader(graph_list, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
    model = GINENet(node_feature_size=num_atom_types, edge_feature_size=1, hidden_dim=256, num_targets=1, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    best_rmse = float('inf')
    patience = 20
    counter = 0
    best_model_state = None
    epochs = 300
    for epoch in range(epochs):
        train_rmse, train_mae, train_mse = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, test_loader, device)
        val_rmse_log = val_metrics['rmse_log']
        if np.isnan(val_rmse_log):
            print("NaN detected in RMSE, stopping training.")
            break
        if val_rmse_log < best_rmse:
            best_rmse = val_rmse_log
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d}: Train RMSE = {train_rmse:.4f}, Val RMSE (log) = {val_rmse_log:.4f}")
        if counter >= patience:
            print("Early stopping triggered.")
            break
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    fold_metrics = evaluate(model, test_loader, device)
    for key in metrics:
        metrics[key].append(fold_metrics[key])
    print(f"Fold {fold + 1} metrics (log):   RMSE={fold_metrics['rmse_log']:.4f}, MAE={fold_metrics['mae_log']:.4f}, "
          f"MSE={fold_metrics['mse_log']:.4f}, R2={fold_metrics['r2_log']:.4f}")
    print(f"Fold {fold + 1} metrics (orig):  RMSE={fold_metrics['rmse_orig']:.4f}, MAE={fold_metrics['mae_orig']:.4f}, "
          f"MSE={fold_metrics['mse_orig']:.4f}, R2={fold_metrics['r2_orig']:.4f}")
    print(f"Fold {fold + 1} completed in {(time.time() - fold_start_time):.2f} seconds.")

# --- Final aggregated results ---
print("\nFinal 5-Fold Cross-Validation Results (mean ± std):")
for key in ['rmse_log', 'mae_log', 'mse_log', 'r2_log']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (log scale)")
for key in ['rmse_orig', 'mae_orig', 'mse_orig', 'r2_orig']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (original scale)")
print(f"\nTotal execution time: {(time.time() - overall_start_time):.2f} seconds")

