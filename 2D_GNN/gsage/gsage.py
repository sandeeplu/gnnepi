import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from math import sqrt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_smiles
# Define GraphSAGE Model
from torch_geometric.nn import SAGEConv

from matplotlib.offsetbox import AnchoredText
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set up
start_time = time.time()
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

# Data Loading and Log Transform
# --------------------------
epsilon = 1e-6  # Small constant to avoid log(0)
df = pd.read_csv('../../../epi.csv')  # Use actual path

# Log-transform the target
df["log_electrophilicity_index"] = np.log(df["electrophilicity_index"] + epsilon)

# --------------------------
# Graph Construction
# --------------------------
graph_list = []
for i, smile in enumerate(df['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df['log_electrophilicity_index'][i]], dtype=torch.float)
    graph_list.append(g)

# Ensure consistent node feature shape
assert all(g.x.shape[1] == 9 for g in graph_list), "Inconsistent in_channels!"


# Print the first graph in the list
g = graph_list[1]

print("Graph object:")
print(g)

print("\nNode feature matrix (x):")
print(g.x)

print("\nShape of x (i.e., [num_nodes, in_channels]):", g.x.shape)

print("\nEdge index:")
print(g.edge_index)

print("\nLabel (y):", g.y)

##
assert all(g.x.shape[1] == 9 for g in graph_list), "Inconsistent in_channels!"

# Custom Dataset Class
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return 'Lipophilicity.csv'
    @property
    def processed_file_names(self): return 'data.dt'
    def download(self): pass

    def process(self):
        data_list = graph_list
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list, self.processed_paths[0])

lipo = MyOwnDataset(root='.')

# Dataset splitting
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define GraphSAGE Model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
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

# Define model and optimizer
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGEModel(in_channels=9, hidden_channels=256, out_channels=1, num_layers=4, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00004407086844982349, weight_decay=0.000004856076517711844)

# Training function
def train(loader):
    model.train()
    total_loss = total_mae = total_mse = total_samples = 0
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

# Testing function
@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse), mean_absolute_error(actuals, preds), mse

#Check exit dir

os.makedirs('models', exist_ok=True)

# Training loop with Early Stopping
best_rmse = float('inf')
epochs = 300
patience = 20
counter = 0
model.reset_parameters()
train_rmse, train_mae, train_mse = [], [], []
test_rmse, test_mae, test_mse = [], [], []

for epoch in range(epochs):
    tr_rmse, tr_mae, tr_mse = train(train_loader)
    te_rmse, te_mae, te_mse = test(test_loader)
    train_rmse.append(tr_rmse)
    train_mae.append(tr_mae)
    train_mse.append(tr_mse)
    test_rmse.append(te_rmse)
    test_mae.append(te_mae)
    test_mse.append(te_mse)
    print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {tr_rmse:.4f}, Test RMSE: {te_rmse:.4f}")

    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break


# Load pretrained model weights if available
if os.path.exists("models/best_model.pth"):
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.to(device)
    model.eval()

# --------------------------
# Evaluation Function
# --------------------------
@torch.no_grad()
def evaluate(loader):
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
    actuals_log = np.concatenate(actuals_log)
    preds_orig = np.concatenate(preds_orig)
    actuals_orig = np.concatenate(actuals_orig)

    df_eval = pd.DataFrame({
        'pred_log': preds_log,
        'actual_log': actuals_log,
        'pred_orig': preds_orig,
        'actual_orig': actuals_orig
    })

    r2_log = r2_score(actuals_log, preds_log)
    mse_log = mean_squared_error(actuals_log, preds_log)
    mae_log = mean_absolute_error(actuals_log, preds_log)

    r2_orig = r2_score(actuals_orig, preds_orig)
    mse_orig = mean_squared_error(actuals_orig, preds_orig)
    mae_orig = mean_absolute_error(actuals_orig, preds_orig)

    return df_eval, (r2_log, mse_log, mae_log), (r2_orig, mse_orig, mae_orig)

# --------------------------
# Model Evaluation
# --------------------------
train_results, (train_r2_log, train_mse_log, train_mae_log), (train_r2_orig, train_mse_orig, train_mae_orig) = evaluate(train_loader)
test_results, (test_r2_log, test_mse_log, test_mae_log), (test_r2_orig, test_mse_orig, test_mae_orig) = evaluate(test_loader)

print("\n\U0001F4D8 Final Performance Metrics")
print("-" * 50)
print("Log Scale Evaluation (model output space)")
print(f"Train MAE   : {train_mae_log:.4f}")
print(f"Test  MAE   : {test_mae_log:.4f}")
print(f"Train MSE   : {train_mse_log:.4f}")
print(f"Test  MSE   : {test_mse_log:.4f}")
print(f"Train RMSE  : {np.sqrt(train_mse_log):.4f}")
print(f"Test  RMSE  : {np.sqrt(test_mse_log):.4f}")
print(f"Train R²    : {train_r2_log:.4f}")
print(f"Test  R²    : {test_r2_log:.4f}")
print("-" * 50)
print("Original Scale Evaluation (after exp transform)")
print(f"Train MAE   : {train_mae_orig:.4f}")
print(f"Test  MAE   : {test_mae_orig:.4f}")
print(f"Train MSE   : {train_mse_orig:.4f}")
print(f"Test  MSE   : {test_mse_orig:.4f}")
print(f"Train RMSE  : {np.sqrt(train_mse_orig):.4f}")
print(f"Test  RMSE  : {np.sqrt(test_mse_orig):.4f}")
print(f"Train R²    : {train_r2_orig:.4f}")
print(f"Test  R²    : {test_r2_orig:.4f}")
print("-" * 50)

print(f"Execution time: {(time.time() - start_time)/60:.2f} minutes")
test_results.to_csv("predictions_log_and_original.csv", index=False)


# ------- PLOTTING SECTION --------
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2
})

os.makedirs('plots', exist_ok=True)

###Epoch vs RMSE
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', color='red')
plt.plot(range(1, len(test_rmse)+1), test_rmse, label='Test RMSE', color='blue')
plt.xlabel('Epoch'); plt.ylabel('RMSE')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_rmse.png'); plt.close()


# ---------- LOG SCALE: Actual vs Predicted & Residuals -----------
plt.figure(figsize=(14, 6), dpi=300)
# -- Actual vs Predicted (log)
plt.subplot(1, 2, 1)
plt.scatter(
    test_results['actual_log'], test_results['pred_log'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.regplot(
    data=train_results, x='actual_log', y='pred_log', color='red',
    scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'}
)
plt.xlabel('Actual (log)', fontsize=20, fontweight='bold')
plt.ylabel('Predicted (log)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=20)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_log:.3f}\nR² (Test): {test_r2_log:.3f}",
    loc='upper left', prop=dict(size=18, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

# -- Residuals (log)
plt.subplot(1, 2, 2)
plt.scatter(
    test_results['actual_log'], test_results['actual_log'] - test_results['pred_log'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.scatterplot(
    x=train_results['actual_log'],
    y=train_results['actual_log'] - train_results['pred_log'],
    color='red', alpha=0.3, edgecolor='black', label='Train'
)
plt.xlabel('Actual (log)', fontsize=20, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(loc='upper left', frameon=False, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'GSAGE',
    transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_log_hd.png', dpi=600)
plt.close()

# ---------- LOG SCALE: KDE Histogram of Errors -----------
plt.figure(figsize=(7, 6), dpi=300)
sns.histplot(
    test_results['actual_log'] - test_results['pred_log'],
    kde=True, color='blue', bins=30, edgecolor='black'
)
plt.xlabel('Prediction Error (log)', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/error_histogram_kde_log_hd.png', dpi=600)
plt.close()

# ---------- ORIGINAL SCALE: Actual vs Predicted & Residuals -----------
plt.figure(figsize=(14, 6), dpi=300)
# -- Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(
    test_results['actual_orig'], test_results['pred_orig'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.regplot(
    data=train_results, x='actual_orig', y='pred_orig', color='red',
    scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'}
)
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Predicted', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=20)
anchored_text = AnchoredText(
    f"R² (Train): {train_r2_orig:.3f}\nR² (Test): {test_r2_orig:.3f}",
    loc='upper left', prop=dict(size=18, weight='bold')
)
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

# -- Residuals
plt.subplot(1, 2, 2)
plt.scatter(
    test_results['actual_orig'], test_results['actual_orig'] - test_results['pred_orig'],
    color='blue', label='Test', edgecolors='black', s=50
)
sns.scatterplot(
    x=train_results['actual_orig'],
    y=train_results['actual_orig'] - train_results['pred_orig'],
    color='red', alpha=0.3, edgecolor='black', label='Train'
)
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(loc='upper left', frameon=False, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.text(
    0.99, 0.03, 'GSAGE',
    transform=ax.transAxes,
    fontsize=18, fontweight='bold', color='black',
    va='bottom', ha='right', alpha=0.9
)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_original_hd.png', dpi=600)
plt.close()

# ---------- ORIGINAL SCALE: KDE Histogram of Errors -----------
plt.figure(figsize=(7, 6), dpi=300)
sns.histplot(
    test_results['actual_orig'] - test_results['pred_orig'],
    kde=True, color='blue', bins=30, edgecolor='black'
)
plt.xlabel('Prediction Error', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/error_histogram_kde_original_hd.png', dpi=600)
plt.close()
