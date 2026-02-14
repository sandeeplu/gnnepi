import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState
from rdkit import Chem
from rdkit.Chem import AllChem

# ======== Reproducibility ========
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

# ======== Feature Extraction ========
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic())
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    bond_type = [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC)
    ]
    return bond_type + [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo())
    ]

def bond_index_and_attrs(mol):
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edges.extend([[i, j], [j, i]])
        attrs.extend([bf, bf])
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(attrs, dtype=torch.float)

# ======== Data Preparation ========
def prepare_data(data_file, target_col):
    df = pd.read_csv(data_file)
    epsilon = 1e-6
    if target_col not in df.columns and 'electrophilicity_index' in df.columns:
        df[target_col] = np.log(df['electrophilicity_index'] + epsilon)
    graph_list = []
    for i, smile in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smile)
        if mol is None: continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0: continue
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        feats, pos = [], []
        for atom in mol.GetAtoms():
            feats.append(atom_features(atom))
            c = conf.GetAtomPosition(atom.GetIdx())
            pos.append([c.x, c.y, c.z])
        x = torch.tensor(feats, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        edge_index, edge_attr = bond_index_and_attrs(mol)
        y_val = df[target_col].iloc[i]
        if np.isnan(y_val) or np.isinf(y_val):
            continue
        y = torch.tensor([y_val], dtype=torch.float)
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y))
    return graph_list

# ======== GemNet-Inspired Model ========
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class SimpleGemNetLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout):
        super().__init__(aggr='add')
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        x_emb = self.node_emb(x)
        edge_emb = self.edge_emb(edge_attr)
        row, col = edge_index
        edge_dist = torch.norm(pos[row] - pos[col], dim=1).unsqueeze(-1)
        return self.propagate(edge_index, x=x_emb, edge_attr=edge_emb, edge_dist=edge_dist)

    def message(self, x_i, x_j, edge_attr, edge_dist):
        return self.mlp_msg(torch.cat([x_i, x_j, edge_dist], dim=-1)) + edge_attr

    def update(self, aggr_out):
        return self.mlp_update(aggr_out)

class SimpleGemNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([SimpleGemNetLayer(node_dim, edge_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, pos)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = scatter(x, batch, dim=0, reduce='add')
        return self.fc(x)

# ======== Train/Test ========
def train(model, loader, optimizer, device):
    model.train()
    loss_sum, samples = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y.view(-1,1))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * data.num_graphs
        samples += data.num_graphs
    return np.sqrt(loss_sum / samples)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds); actuals = np.concatenate(actuals)
    # Log-scale metrics
    mae_log = mean_absolute_error(actuals, preds)
    mse_log = mean_squared_error(actuals, preds)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(actuals, preds)
    # Original scale metrics
    preds_orig, actuals_orig = np.exp(preds), np.exp(actuals)
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(actuals_orig, preds_orig)
    return mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig

# ======== Optuna Objective ========
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    num_layers = trial.suggest_int("num_layers", 3, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    epochs = trial.suggest_int("epochs", 120, 300, step=60)

    graph_list = prepare_data("epi.csv", "log_electrophilicity_index")
    train_size = int(0.8 * len(graph_list))
    test_size = len(graph_list) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim, edge_dim = graph_list[0].x.shape[1], graph_list[0].edge_attr.shape[1]
    model = SimpleGemNet(node_dim, edge_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        train(model, train_loader, optimizer, device)
        metrics = test(model, test_loader, device)
        trial.report(metrics[2], epoch)  # report log-scale RMSE
        if trial.should_prune():
            raise optuna.TrialPruned()

    mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig = metrics
    trial.set_user_attr("mae_log", mae_log)
    trial.set_user_attr("mse_log", mse_log)
    trial.set_user_attr("r2_log", r2_log)
    trial.set_user_attr("mae_orig", mae_orig)
    trial.set_user_attr("mse_orig", mse_orig)
    trial.set_user_attr("rmse_orig", rmse_orig)
    trial.set_user_attr("r2_orig", r2_orig)
    return rmse_log

# ======== Run Study ========
def run_optimization():
    study = optuna.create_study(direction='minimize',
        study_name='hyperparameter-tune-gemnet',
        storage='sqlite:///hyperparameter-tune-gemnet.db',
        load_if_exists=True)
    study.optimize(objective, n_trials=50)
    print("\nBest RMSE (log):", study.best_value)
    print("Best Params:", study.best_trial.params)
    print("Best Metrics (log scale):",
          "RMSE:", study.best_value,
          "MAE:", study.best_trial.user_attrs["mae_log"],
          "R²:", study.best_trial.user_attrs["r2_log"])
    print("Best Metrics (original scale):",
          "RMSE:", study.best_trial.user_attrs["rmse_orig"],
          "MAE:", study.best_trial.user_attrs["mae_orig"],
          "R²:", study.best_trial.user_attrs["r2_orig"])

if __name__ == "__main__":
    seed_set(42)
    run_optimization()

