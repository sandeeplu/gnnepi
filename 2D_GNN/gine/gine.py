import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn import Sequential as Seq, Linear, BatchNorm1d, ReLU
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

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
df = pd.read_csv('../epi.csv')
df["log_electrophilicity_index"] = np.log(df["electrophilicity_index"] + epsilon)

# --- Atom types for one-hot encoding (QM9 only!) ---
atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
atom_type_to_idx = {anum: idx for idx, anum in enumerate(atom_types)}
num_atom_types = len(atom_types)

# --- Graph Construction for GINE/MPNN ---
graph_list = []
for i, smile in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0:
        continue
    AllChem.UFFOptimizeMolecule(mol)
    # Node features: one-hot encoding of atom types
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [0] * num_atom_types
        anum = atom.GetAtomicNum()
        if anum in atom_type_to_idx:
            feat[atom_type_to_idx[anum]] = 1
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)
    # Edge index and edge_attr (bond type as edge_attr)
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

# --- Dataset Splitting ---
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- GINE/MPNN Model ---
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
        # Global pooling for each layer
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)
        h4_pool = global_add_pool(h4, batch)
        # Concatenate graph-level representations
        h = torch.cat([h1_pool, h2_pool, h3_pool, h4_pool], dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        return h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINENet(node_feature_size=num_atom_types, edge_feature_size=1, hidden_dim=256, num_targets=1, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

# --- Training Function with Gradient Clipping and NaN Checks ---
def train(loader):
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

# --- Testing Function with NaN Checks ---
@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        if torch.isnan(out).any() or torch.isinf(out).any():
            continue
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    if len(preds) == 0 or len(actuals) == 0:
        return float('nan'), float('nan'), float('nan')
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    if np.isnan(preds).any() or np.isnan(actuals).any():
        print("NaN in predictions or actuals during evaluation.")
        return float('nan'), float('nan'), float('nan')
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse), mean_absolute_error(actuals, preds), mse

# --- Training Loop with Early Stopping ---
os.makedirs('models', exist_ok=True)
best_rmse = float('inf')
epochs = 300
patience = 20
counter = 0
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

    if np.isnan(te_rmse) or np.isnan(tr_rmse):
        print("NaN detected in RMSE, stopping training.")
        break

    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), 'models/best_ginenet_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --- Load Best Model for Evaluation ---
if os.path.exists("models/best_ginenet_model.pth"):
    model.load_state_dict(torch.load("models/best_ginenet_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(loader):
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

# --- Model Evaluation ---
train_results, (train_r2_log, train_mse_log, train_mae_log), (train_r2_orig, train_mse_orig, train_mae_orig) = evaluate(train_loader)
test_results, (test_r2_log, test_mse_log, test_mae_log), (test_r2_orig, test_mse_orig, test_mae_orig) = evaluate(test_loader)

print("\n Final Performance Metrics")
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
test_results.to_csv("ginenet_predictions_log_and_original.csv", index=False)


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

# RMSE over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', color='red')
plt.plot(range(1, len(test_rmse)+1), test_rmse, label='Test RMSE', color='blue')
plt.xlabel('Epoch'); plt.ylabel('RMSE')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_rmse_ginenet.png'); plt.close()



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
    f"R² (Train): {train_r2_log:.4f}\nR² (Test): {test_r2_log:.4f}",
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
    0.99, 0.03, 'GINE',
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
    f"R² (Train): {train_r2_orig:.4f}\nR² (Test): {test_r2_orig:.4f}",
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
    0.99, 0.03, 'GINE',
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
