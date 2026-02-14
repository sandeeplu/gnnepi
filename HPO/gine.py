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
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn import Sequential as Seq, Linear, BatchNorm1d, ReLU

# Seed setup for reproducibility
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data preparation function
def prepare_data(data_file, target_col):
    epsilon = 1e-6
    df = pd.read_csv(data_file)
    if target_col not in df.columns and 'electrophilicity_index' in df.columns:
        df[target_col] = np.log(df['electrophilicity_index'] + epsilon)
        
    atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
    atom_type_to_idx = {anum: idx for idx, anum in enumerate(atom_types)}
    num_atom_types = len(atom_types)

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
            edge_index.extend([[i1, i2], [i2, i1]])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attr.extend([[bond_type], [bond_type]])
        if len(edge_index) == 0:
            continue
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        y_val = df[target_col].iloc[i]
        if np.isnan(y_val) or np.isinf(y_val):
            continue
        y = torch.tensor([y_val], dtype=torch.float)
        
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graph_list, num_atom_types

# GINE model definition
class GINENet(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_feature_size if i == 0 else hidden_dim
            nn_layer = Seq(
                Linear(in_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_layer, edge_dim=edge_feature_size))
        
        self.lin1 = Linear(hidden_dim * num_layers, hidden_dim * num_layers)
        self.lin2 = Linear(hidden_dim * num_layers, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        outs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            outs.append(global_add_pool(h, batch))
        h = torch.cat(outs, dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

# Training function
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return np.sqrt(total_loss / total_samples)

# Evaluation function with log and original scale metrics
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        targets.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    # Log scale metrics
    mae_log = mean_absolute_error(targets, preds)
    mse_log = mean_squared_error(targets, preds)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(targets, preds)
    
    # Original scale metrics
    preds_orig = np.exp(preds)
    targets_orig = np.exp(targets)
    mae_orig = mean_absolute_error(targets_orig, preds_orig)
    mse_orig = mean_squared_error(targets_orig, preds_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(targets_orig, preds_orig)
    
    return mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig

# Objective function for Optuna
def objective(trial):
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    epochs = trial.suggest_int("epochs", 120, 300, step=60)

    graph_list, num_atom_types = prepare_data("epi.csv", "log_electrophilicity_index")  # Use proper path and col
    seed_set(42)
    train_size = int(0.8 * len(graph_list))
    test_size = len(graph_list) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINENet(num_atom_types, 1, hidden_dim, num_layers, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_rmse = float("inf")
    patience = 15
    counter = 0
    
    for epoch in range(epochs):
        train_rmse = train(model, train_loader, optimizer, device)
        mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig = evaluate(model, test_loader, device)
        
        trial.report(rmse_log, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if rmse_log < best_rmse:
            best_rmse = rmse_log
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    trial.set_user_attr("mae_log", mae_log)
    trial.set_user_attr("mse_log", mse_log)
    trial.set_user_attr("rmse_log", rmse_log)
    trial.set_user_attr("r2_log", r2_log)
    trial.set_user_attr("mae_orig", mae_orig)
    trial.set_user_attr("mse_orig", mse_orig)
    trial.set_user_attr("rmse_orig", rmse_orig)
    trial.set_user_attr("r2_orig", r2_orig)
    
    print(f"Trial done - RMSE log: {rmse_log:.4f}, RMSE original: {rmse_orig:.4f}")
    
    return best_rmse

# Run the optimization study
def run_optimization():
    seed_set(42)
    study = optuna.create_study(
        direction="minimize",
        study_name="gine_optimization",
        storage="sqlite:///gine_optimization.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=60)
    
    best = study.best_trial
    print("\nBest Trial Metrics:")
    print(f"  MAE (log scale): {best.user_attrs['mae_log']:.4f}")
    print(f"  MSE (log scale): {best.user_attrs['mse_log']:.4f}")
    print(f"  RMSE (log scale): {best.user_attrs['rmse_log']:.4f}")
    print(f"  R2  (log scale): {best.user_attrs['r2_log']:.4f}")
    print(f"  MAE (original): {best.user_attrs['mae_orig']:.4f}")
    print(f"  MSE (original): {best.user_attrs['mse_orig']:.4f}")
    print(f"  RMSE (original): {best.user_attrs['rmse_orig']:.4f}")
    print(f"  R2  (original): {best.user_attrs['r2_orig']:.4f}")
    
    print("\nBest hyperparameters:")
    print(best.params)
    
    print("\nAll Completed Trials:")
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            print(f"Trial {trial.number}: RMSE (log) = {trial.value:.4f}, RMSE (orig) = {trial.user_attrs['rmse_orig']:.4f}, Params = {trial.params}")

if __name__ == "__main__":
    run_optimization()

