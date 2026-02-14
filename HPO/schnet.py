import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.trial import TrialState
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Seed Setup ---
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Data Preparation ---
def prepare_data(data_file, target_col="log_electrophilicity_index"):
    epsilon = 1e-6
    df = pd.read_csv(data_file)
    if target_col not in df.columns and 'electrophilicity_index' in df.columns:
        df[target_col] = np.log(df['electrophilicity_index'] + epsilon)

    graph_list = []
    failed_smiles = []

    for i, smile in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            failed_smiles.append(smile)
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            failed_smiles.append(smile)
            continue
        AllChem.UFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()

        z = []
        pos = []
        for atom in mol.GetAtoms():
            z.append(atom.GetAtomicNum())
            p = conformer.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        z = torch.tensor(z, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)

        y_val = df[target_col].iloc[i]
        if np.isnan(y_val) or np.isinf(y_val):
            failed_smiles.append(smile)
            continue
        y = torch.tensor([y_val], dtype=torch.float)
        graph_list.append(Data(z=z, pos=pos, y=y))

    return graph_list

# --- Training ---
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = total_samples = 0
    for data in train_loader:
        data = data.to(device)
        if torch.isnan(data.y).any() or torch.isinf(data.y).any():
            continue
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        if torch.isnan(out).any() or torch.isinf(out).any():
            continue
        loss = F.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    if total_samples == 0:
        return float('nan')
    return np.sqrt(total_loss / total_samples)

# --- Testing ---
@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    preds, actuals = [], []
    for data in test_loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch)
        if torch.isnan(out).any() or torch.isinf(out).any():
            continue
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    if len(preds) == 0 or len(actuals) == 0:
        return (float('nan'),) * 8
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    # Compute metrics on log scale
    mae_log = mean_absolute_error(actuals, preds)
    mse_log = mean_squared_error(actuals, preds)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(actuals, preds)
    # Compute metrics on original scale
    preds_orig = np.exp(preds)
    actuals_orig = np.exp(actuals)
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(actuals_orig, preds_orig)
    return mae_log, mse_log, rmse_log, r2_log, mae_orig, mse_orig, rmse_orig, r2_orig

# --- Optuna Objective ---
def objective(trial):
    # Hyperparameters
    hidden_channels = trial.suggest_int("hidden_channels", 64, 256, step=64)
    num_interactions = trial.suggest_int("num_interactions", 3, 6)
    num_gaussians = trial.suggest_categorical("num_gaussians", [25, 50, 100])
    cutoff = trial.suggest_float("cutoff", 5.0, 12.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    epochs = trial.suggest_int("epochs", 100, 300, step=50)

    graph_list = prepare_data('epi.csv', "log_electrophilicity_index")
    seed_set(42)
    train_ratio = 0.8
    train_size = int(train_ratio * len(graph_list))
    test_size = len(graph_list) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(hidden_channels=hidden_channels, num_filters=hidden_channels,
                   num_interactions=num_interactions, num_gaussians=num_gaussians,
                   cutoff=cutoff, readout='add').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float('inf')
    patience = 15
    counter = 0

    for epoch in range(epochs):
        train_rmse = train(model, train_loader, optimizer, device)
        mae_log, mse_log, test_rmse, test_r2, mae_orig, mse_orig, rmse_orig, r2_orig = test(model, test_loader, device)
        trial.report(test_rmse, epoch)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("mae_log", mae_log)
    trial.set_user_attr("mse_log", mse_log)
    trial.set_user_attr("r2_log", test_r2)
    trial.set_user_attr("mae_orig", mae_orig)
    trial.set_user_attr("mse_orig", mse_orig)
    trial.set_user_attr("rmse_orig", rmse_orig)
    trial.set_user_attr("r2_orig", r2_orig)

    print(f"Trial RMSE (log): {best_rmse:.4f}, RMSE (orig): {rmse_orig:.4f}, MAE (log): {mae_log:.4f}, MAE (orig): {mae_orig:.4f}, R2 (log): {test_r2:.4f}, R2 (orig): {r2_orig:.4f}")
    return best_rmse

# --- Run the Optuna Study ---
def run_optimization():
    seed_set(42)
    start_time = time.time()
    study = optuna.create_study(
        direction='minimize',
        study_name='hyperparameter-tune-schnet',
        storage='sqlite:///hyperparameter-tune-schnet.db',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)
    end_time = time.time()

    print(f"Optimization Duration: {end_time - start_time:.2f} seconds")
    print("Best RMSE (log scale):", study.best_value)
    print("Best Params:", study.best_trial.params)

    best_trial = study.best_trial
    print("\nBest Trial Metrics:")
    print(f"  MAE (log): {best_trial.user_attrs['mae_log']:.4f}")
    print(f"  MSE (log): {best_trial.user_attrs['mse_log']:.4f}")
    print(f"  R2  (log): {best_trial.user_attrs['r2_log']:.4f}")
    print(f"  MAE (orig): {best_trial.user_attrs['mae_orig']:.4f}")
    print(f"  MSE (orig): {best_trial.user_attrs['mse_orig']:.4f}")
    print(f"  RMSE(orig): {best_trial.user_attrs['rmse_orig']:.4f}")
    print(f"  R2  (orig): {best_trial.user_attrs['r2_orig']:.4f}")

    print("\n=== Completed Trials Summary ===")
    for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
        print(f"Trial {t.number}: RMSE_log={t.value:.4f}, RMSE_orig={t.user_attrs.get('rmse_orig', float('nan')):.4f}, Params={t.params}")

if __name__ == "__main__":
    run_optimization()

