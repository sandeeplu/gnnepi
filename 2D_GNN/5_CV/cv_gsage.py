import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import SubsetRandomSampler
from torch_geometric.utils import from_smiles

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

# --- Graph Construction ---
graph_list = []
for i, smile in enumerate(df['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df['log_electrophilicity_index'][i]], dtype=torch.float)
    graph_list.append(g)
assert all(g.x.shape[1] == 9 for g in graph_list), "Inconsistent in_channels!"

# --- GraphSAGE Model ---
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return global_mean_pool(x, batch)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

# --- Training function ---
def train(model, loader, optimizer, device):
    model.train()
    total_loss = total_mae = total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        mse_loss = F.mse_loss(out, data.y.view(-1, 1))
        mae_loss = F.l1_loss(out, data.y.view(-1, 1))
        mse_loss.backward()
        optimizer.step()
        total_loss += mse_loss.item() * data.num_graphs
        total_mae += mae_loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return np.sqrt(total_loss / total_samples), total_mae / total_samples, total_loss / total_samples

# --- Evaluation function ---
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_log, actuals_log = [], []
    preds_orig, actuals_orig = [], []
    for data in loader:
        data = data.to(device)
        out_log = model(data.x, data.edge_index, data.batch).view(-1)
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
    train_loader = DataLoader(graph_list, batch_size=32, sampler=SubsetRandomSampler(train_idx))
    test_loader = DataLoader(graph_list, batch_size=32, sampler=SubsetRandomSampler(test_idx))
    model = GraphSAGEModel(in_channels=9, hidden_channels=192, out_channels=1, num_layers=4, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004407086844982349, weight_decay=0.000004856076517711844)
    model.reset_parameters()
    best_rmse = float('inf')
    patience = 20
    counter = 0
    best_model_state = None
    epochs = 300
    for epoch in range(epochs):
        train_rmse, train_mae, train_mse = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, test_loader, device)
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

