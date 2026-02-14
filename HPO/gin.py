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
from torch_geometric.nn import GINConv, MLP, global_mean_pool
from torch_geometric.utils import from_smiles
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState

# Set seed for reproducibility
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data preparation function with log transform
def prepare_data(data_file, target_col):
    epsilon = 1e-6
    df = pd.read_csv(data_file)
    if target_col not in df.columns and 'electrophilicity_index' in df.columns:
        df[target_col] = np.log(df['electrophilicity_index'] + epsilon)
    graph_list = []
    for i, smile in enumerate(df['smiles']):
        g = from_smiles(smile)
        g.x = g.x.float()
        y_val = df[target_col].iloc[i]
        if np.isnan(y_val) or np.isinf(y_val):
            continue
        g.y = torch.tensor([y_val], dtype=torch.float)
        graph_list.append(g)
    return graph_list

# Define the GIN model
class GINNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GINNet, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        
        for i in range(num_layers):
            if i == 0:
                mlp = MLP([in_channels, hidden_channels, hidden_channels])
            else:
                mlp = MLP([hidden_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(mlp, train_eps=False))
        self.out_mlp = MLP([hidden_channels, hidden_channels, out_channels], dropout=dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_mlp.reset_parameters()

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch)
        out = self.out_mlp(h)
        return out

# Training function
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, data.y.view(-1,1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return np.sqrt(total_loss / total_samples)

# Evaluation function returning both log and original scale metrics
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    for data in dataloader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        targets.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Metrics on log scale
    mae_log = mean_absolute_error(targets, preds)
    mse_log = mean_squared_error(targets, preds)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(targets, preds)

    # Metrics on original scale
    preds_orig = np.exp(preds)
    targets_orig = np.exp(targets)
    mae_orig = mean_absolute_error(targets_orig, preds_orig)
    mse_orig = mean_squared_error(targets_orig, preds_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(targets_orig, preds_orig)

    return mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig

# Optuna objective function
def objective(trial):
    # Hyperparameters search space
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    hidden_channels = trial.suggest_int("hidden_channels", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    epochs = trial.suggest_int("epochs", 120, 300, step=60)

    graph_list = prepare_data('epi.csv', 'log_electrophilicity_index')  # Replace with your correct path and target column
    seed_set(42)
    train_size = int(len(graph_list) * 0.8)
    test_size = len(graph_list) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = graph_list[0].x.shape[1]
    model = GINNet(in_channels, hidden_channels, 1, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float('inf')
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

    trial.set_user_attr('mae_log', mae_log)
    trial.set_user_attr('mse_log', mse_log)
    trial.set_user_attr('rmse_log', rmse_log)
    trial.set_user_attr('r2_log', r2_log)
    trial.set_user_attr('mae_orig', mae_orig)
    trial.set_user_attr('mse_orig', mse_orig)
    trial.set_user_attr('rmse_orig', rmse_orig)
    trial.set_user_attr('r2_orig', r2_orig)

    print(f'Trial complete: RMSE (log): {rmse_log:.4f}, RMSE (orig): {rmse_orig:.4f}')
    return best_rmse

# Run the optimization
def run_optimization():
    seed_set(42)
    study = optuna.create_study(
        direction='minimize',
        study_name='gin_optimization',
        storage='sqlite:///gin_optimization.db',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)

    best = study.best_trial
    print('\nBest Trial Metrics:')
    print(f"  MAE (log): {best.user_attrs['mae_log']:.4f}")
    print(f"  MSE (log): {best.user_attrs['mse_log']:.4f}")
    print(f"  RMSE (log): {best.user_attrs['rmse_log']:.4f}")
    print(f"  R2 (log): {best.user_attrs['r2_log']:.4f}")
    print(f"  MAE (orig): {best.user_attrs['mae_orig']:.4f}")
    print(f"  MSE (orig): {best.user_attrs['mse_orig']:.4f}")
    print(f"  RMSE (orig): {best.user_attrs['rmse_orig']:.4f}")
    print(f"  R2 (orig): {best.user_attrs['r2_orig']:.4f}")

    print('\nBest Hyperparameters:')
    print(best.params)

    print('\nAll Completed Trials:')
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            print(f'Trial {trial.number}: RMSE (log)={trial.value:.4f}, Params={trial.params}')


if __name__ == '__main__':
    run_optimization()

