import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import SubsetRandomSampler
from torch_geometric.nn import AttentiveFP

# Set reproducibility
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

# Load your dataset (adjust path accordingly)
epsilon = 1e-6
df = pd.read_csv('../../../../epi.csv')
df["log_electrophilicity_index"] = np.log(df["electrophilicity_index"] + epsilon)

# Prepare graph list
graph_list = []
for i, smile in enumerate(df['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df['log_electrophilicity_index'][i]], dtype=torch.float)
    graph_list.append(g)

# Assert consistent node feature size
assert all(g.x.shape[1] == 9 for g in graph_list), "Inconsistent in_channels!"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    rmse = np.sqrt(total_loss / len(loader.dataset))
    return rmse

# Evaluation function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_log, targets_log = [], []
    for data in loader:
        data = data.to(device)
        out_log = model(data.x, data.edge_index, data.edge_attr, data.batch).view(-1)
        y_log = data.y.view(-1)
        preds_log.append(out_log.cpu().numpy())
        targets_log.append(y_log.cpu().numpy())
    preds_log = np.concatenate(preds_log)
    targets_log = np.concatenate(targets_log)

    # Metrics in log space
    rmse_log = np.sqrt(mean_squared_error(targets_log, preds_log))
    mae_log = mean_absolute_error(targets_log, preds_log)
    mse_log = mean_squared_error(targets_log, preds_log)
    r2_log = r2_score(targets_log, preds_log)

    # Metrics in original space
    preds_orig = np.exp(preds_log)
    targets_orig = np.exp(targets_log)
    rmse_orig = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    mae_orig = mean_absolute_error(targets_orig, preds_orig)
    mse_orig = mean_squared_error(targets_orig, preds_orig)
    r2_orig = r2_score(targets_orig, preds_orig)

    return {
        'rmse_log': rmse_log, 'mae_log': mae_log, 'mse_log': mse_log, 'r2_log': r2_log,
        'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'mse_orig': mse_orig, 'r2_orig': r2_orig
    }

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

metrics = {
    'rmse_log': [], 'mae_log': [], 'mse_log': [], 'r2_log': [],
    'rmse_orig': [], 'mae_orig': [], 'mse_orig': [], 'r2_orig': []
}
overall_start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(kfold.split(graph_list)):
    print(f"\nFOLD {fold + 1} ----------------------------")
    fold_start_time = time.time()

    train_loader = DataLoader(graph_list, batch_size=128, sampler=SubsetRandomSampler(train_idx))
    test_loader = DataLoader(graph_list, batch_size=128, sampler=SubsetRandomSampler(test_idx))
    model = AttentiveFP(in_channels=9, hidden_channels=256, out_channels=1,
                        edge_dim=3, num_layers=5, num_timesteps=2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    model.reset_parameters()

    best_rmse = float('inf')
    patience = 20
    counter = 0
    best_model_state = None

    epochs = 300
    for epoch in range(epochs):
        train_rmse = train(model, train_loader, optimizer)
        val_metrics = evaluate(model, test_loader)
        val_rmse_log = val_metrics['rmse_log']

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
    fold_metrics = evaluate(model, test_loader)
    for key in metrics:
        metrics[key].append(fold_metrics[key])

    print(f"Fold {fold + 1} metrics (log):   RMSE={fold_metrics['rmse_log']:.4f}, MAE={fold_metrics['mae_log']:.4f}, "
          f"MSE={fold_metrics['mse_log']:.4f}, R2={fold_metrics['r2_log']:.4f}")
    print(f"Fold {fold + 1} metrics (orig):  RMSE={fold_metrics['rmse_orig']:.4f}, MAE={fold_metrics['mae_orig']:.4f}, "
          f"MSE={fold_metrics['mse_orig']:.4f}, R2={fold_metrics['r2_orig']:.4f}")
    print(f"Fold {fold + 1} completed in {(time.time() - fold_start_time):.2f} seconds.")

# Final aggregated results
print("\nFinal 5-Fold Cross-Validation Results (mean ± std):")
for key in ['rmse_log', 'mae_log', 'mse_log', 'r2_log']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (log scale)")
for key in ['rmse_orig', 'mae_orig', 'mse_orig', 'r2_orig']:
    print(f"{key.upper():<10}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f} (original scale)")
print(f"\nTotal execution time: {(time.time() - overall_start_time):.2f} seconds")

