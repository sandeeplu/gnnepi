import os
import time
import random
import numpy as np
import pandas as pd
import torch
import optuna
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import from_smiles
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from optuna.trial import TrialState

# Set deterministic seed for reproducibility
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Prepare dataset with log-transformed target
def prepare_data(data_file, target_col="log_electrophilicity_index"):
    df = pd.read_csv(data_file)
    epsilon = 1e-6
    if target_col not in df.columns and 'electrophilicity_index' in df.columns:
        df[target_col] = np.log(df['electrophilicity_index'] + epsilon)
    graph_list = []
    for i, smile in enumerate(df['smiles']):
        g = from_smiles(smile)
        g.x = g.x.float()
        g.y = torch.tensor([df[target_col][i]], dtype=torch.float)
        graph_list.append(g)
    return graph_list

# Training routine
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_examples += data.num_graphs
    return np.sqrt(total_loss / total_examples)

# Testing routine with log and original scale metrics
@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    preds, actuals = [], []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    # Log scale metrics
    mae_log = mean_absolute_error(actuals, preds)
    mse_log = mean_squared_error(actuals, preds)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(actuals, preds)

    # Original scale metrics (exponentiate predictions and actuals)
    preds_orig = np.exp(preds)
    actuals_orig = np.exp(actuals)
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(actuals_orig, preds_orig)

    return mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig

# Objective function for Optuna optimization
def objective(trial):
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    num_layers = trial.suggest_int("num_layers", 4, 6)
    hidden_channels = trial.suggest_int("hidden_channels", 128, 256, step=64)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    epochs = trial.suggest_int("epochs", 120, 300, step=60)
    num_timesteps = 2  # fixed, can be tuned if desired

    graph_list = prepare_data('epi.csv', "log_electrophilicity_index")
    seed_set(42)
    train_size = int(0.8 * len(graph_list))
    test_size = len(graph_list) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = graph_list[0].x.shape[1]
    edge_dim = graph_list[0].edge_attr.shape[1] if hasattr(graph_list[0], "edge_attr") else 3

    model = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout
    ).to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float('inf')
    patience = 15
    counter = 0

    for epoch in range(epochs):
        train_rmse = train(model, train_loader, optimizer, device)
        mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig = test(model, test_loader, device)
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
    trial.set_user_attr("r2_log", r2_log)
    trial.set_user_attr("mae_orig", mae_orig)
    trial.set_user_attr("mse_orig", mse_orig)
    trial.set_user_attr("rmse_orig", rmse_orig)
    trial.set_user_attr("r2_orig", r2_orig)

    print(f"Trial completed in {epoch+1} epochs - RMSE_log: {rmse_log:.4f}, RMSE_orig: {rmse_orig:.4f}, MAE_log: {mae_log:.4f}, MAE_orig: {mae_orig:.4f}, R2_log: {r2_log:.4f}, R2_orig: {r2_orig:.4f}")

    return best_rmse

# Run Optuna study
def run_optimization():
    seed_set(42)
    start_time = time.time()
    study = optuna.create_study(
        direction='minimize',
        study_name='hyperparameter_tune_attentivefp',
        storage='sqlite:///attentivefp_optimization.db',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=60)
    end_time = time.time()

    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Best RMSE (log scale): {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    best_trial = study.best_trial
    print(f"Best Trial Metrics:")
    print(f"  MAE (log scale): {best_trial.user_attrs['mae_log']:.4f}")
    print(f"  MSE (log scale): {best_trial.user_attrs['mse_log']:.4f}")
    print(f"  R2  (log scale): {best_trial.user_attrs['r2_log']:.4f}")
    print(f"  MAE (original):  {best_trial.user_attrs['mae_orig']:.4f}")
    print(f"  MSE (original):  {best_trial.user_attrs['mse_orig']:.4f}")
    print(f"  RMSE(original):  {best_trial.user_attrs['rmse_orig']:.4f}")
    print(f"  R2  (original):  {best_trial.user_attrs['r2_orig']:.4f}")

    print("\nAll completed trials:")
    for t in study.trials:
        if t.state == TrialState.COMPLETE:
            print(f"Trial {t.number}: RMSE_log={t.value:.4f}, Params={t.params}")

if __name__ == "__main__":
    run_optimization()

