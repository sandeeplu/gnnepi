import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from rdkit import Chem
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

# === Load CSV and prepare target ===
df = pd.read_csv("epi_133725_3D_exact.csv")
if target_col not in df.columns and "electrophilicity_index" in df.columns:
    df[target_col] = np.log(df["electrophilicity_index"] + epsilon)

# === Atom features function ===
def atom_features(atom):
    Z = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybrid = int(atom.GetHybridization())
    is_aromatic = int(atom.GetIsAromatic())
    return [Z, degree, formal_charge, hybrid, is_aromatic]

# === Build GemNet-style graphs with cutoff edges ===
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
                z_list.append(6)
                x_feats.append([6, 0, 0, 0, 0])

        if not pos_list or not z_list or not x_feats:
            failed.append((smiles, "missing atom/feature/position data"))
            continue

        pos = torch.tensor(pos_list, dtype=torch.float)
        x = torch.tensor(x_feats, dtype=torch.float)

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

        y_val = float(row.get(target_col, np.nan))
        if np.isnan(y_val) or np.isinf(y_val):
            failed.append((smiles, "invalid target"))
            continue
        y = torch.tensor([y_val], dtype=torch.float)

        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph_list.append(data)

    except Exception as e:
        failed.append((row.get("smiles", None), f"exception: {e}"))
        continue

print(f"Graphs built: {len(graph_list)}, Failed: {len(failed)}")

# === GemNet layers and model ===
class RBFLayer(nn.Module):
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
node_dim = graph_list[0].x.shape[1]

# === 5-Fold Cross-Validation ===
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'rmse_log': [], 'mae_log': [], 'mse_log': [], 'r2_log': [],
    'rmse_orig': [], 'mae_orig': [], 'mse_orig': [], 'r2_orig': []
}
epochs = 300
patience = 20
lr = 0.00011953092590523518
weight_decay = 1.8351393227040611e-06

for fold, (train_idx, test_idx) in enumerate(kfold.split(graph_list)):
    print(f"\nFOLD {fold+1} --------------------------")
    train_subset = [graph_list[i] for i in train_idx]
    test_subset  = [graph_list[i] for i in test_idx]
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=generator)
    test_loader  = DataLoader(test_subset, batch_size=32, shuffle=False)

    model = GemNetModel(node_dim=node_dim, hidden_dim=256, num_rbf=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = mae_sum = samples = 0
        for data in train_loader:
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
        train_rmse = np.sqrt(loss_sum / samples)
        train_mae = mae_sum / samples

        model.eval()
        preds, actuals = [], []
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            #preds.append(out.view(-1).cpu().numpy())
            preds.append(out.view(-1).detach().cpu().numpy())
            actuals.append(data.y.view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)
        mse_log = mean_squared_error(actuals, preds)
        rmse_log = np.sqrt(mse_log)

        if rmse_log < best_rmse:
            best_rmse = rmse_log
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print("  Early stopping triggered.")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train RMSE {train_rmse:.4f} | Val RMSE {rmse_log:.4f}")

    # Evaluate best model on test data
    model.load_state_dict(best_model_state)
    model.eval()
    preds_log, actuals_log = [], []
    preds_orig, actuals_orig = [], []
    for data in test_loader:
        data = data.to(device)
        out = model(data).view(-1)
        y = data.y.view(-1)
        #preds_log.append(out.cpu().numpy())
        preds_log.append(out.detach().cpu().numpy())
        actuals_log.append(y.cpu().numpy())
        #preds_orig.append(np.exp(out.cpu().numpy()))
        preds_orig.append(np.exp(out.detach().cpu().numpy()))
        actuals_orig.append(np.exp(y.cpu().numpy()))
    preds_log = np.concatenate(preds_log)
    actuals_log = np.concatenate(actuals_log)
    preds_orig = np.concatenate(preds_orig)
    actuals_orig = np.concatenate(actuals_orig)

    rmse_log = np.sqrt(mean_squared_error(actuals_log, preds_log))
    mae_log = mean_absolute_error(actuals_log, preds_log)
    mse_log = mean_squared_error(actuals_log, preds_log)
    r2_log = r2_score(actuals_log, preds_log)
    rmse_orig = np.sqrt(mean_squared_error(actuals_orig, preds_orig))
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    r2_orig = r2_score(actuals_orig, preds_orig)

    print(f"  Fold {fold+1} log:      RMSE={rmse_log:.4f}, MAE={mae_log:.4f}, MSE={mse_log:.4f}, R2={r2_log:.4f}")
    print(f"  Fold {fold+1} original: RMSE={rmse_orig:.4f}, MAE={mae_orig:.4f}, MSE={mse_orig:.4f}, R2={r2_orig:.4f}")

    for key, val in zip(
        ['rmse_log', 'mae_log', 'mse_log', 'r2_log', 'rmse_orig', 'mae_orig', 'mse_orig', 'r2_orig'],
        [rmse_log, mae_log, mse_log, r2_log, rmse_orig, mae_orig, mse_orig, r2_orig]
    ):
        metrics[key].append(val)

# === Final Aggregate Results ===
print("\nFinal 5-Fold Cross-Validation Results (mean ± std):")
for key in ['rmse_log', 'mae_log', 'mse_log', 'r2_log']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (log scale)")
for key in ['rmse_orig', 'mae_orig', 'mse_orig', 'r2_orig']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (original scale)")

