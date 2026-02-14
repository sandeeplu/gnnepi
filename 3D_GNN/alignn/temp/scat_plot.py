import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

# === Publication-quality plotting style ===
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 16, 'figure.dpi': 300,
    'savefig.dpi': 600, 'axes.linewidth': 1.5, 'lines.linewidth': 2
})

# === Helper functions ===
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def json_to_numpy(json_file):
    """Convert JSON to numpy arrays of actual and predicted values"""
    data = load_json(json_file)
    actual_list, pred_list = [], []
    for entry in data:
        actual_list.extend(entry['target_out'])
        pred = entry['pred_out']
        if isinstance(pred, list):
            pred_list.extend(pred)
        else:
            pred_list.append(pred)
    return np.array(actual_list, dtype=float), np.array(pred_list, dtype=float)

# === Plotting function ===
def plot_actual_vs_pred(train_file, val_file, test_file, plot_dir, scale='log'):
    actual_train, pred_train = json_to_numpy(train_file)
    actual_val, pred_val = json_to_numpy(val_file)
    actual_test, pred_test = json_to_numpy(test_file)

    # Scatter + regression
    plt.figure(figsize=(14, 6), dpi=300)
    plt.subplot(1, 2, 1)
    plt.scatter(actual_test, pred_test, color='blue', s=50, edgecolors='black', label='Test')
    sns.regplot(x=actual_train, y=pred_train, color='red',
                scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'})
    plt.xlabel(f'Actual ({scale})', fontsize=20, fontweight='bold')
    plt.ylabel(f'Predicted ({scale})', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold'); plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=18)

    r2_train = np.corrcoef(actual_train, pred_train)[0,1]**2
    r2_test  = np.corrcoef(actual_test, pred_test)[0,1]**2
    anchored_text = AnchoredText(f"R² (Train): {r2_train:.4f}\nR² (Test): {r2_test:.4f}",
                                 loc='upper left', prop=dict(size=16, weight='bold'))
    plt.gca().add_artist(anchored_text)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Residuals
    plt.subplot(1, 2, 2)
    plt.scatter(actual_test, actual_test - pred_test, color='blue', s=50, edgecolors='black', label='Test')
    plt.scatter(actual_train, actual_train - pred_train, color='red', alpha=0.3, edgecolors='black', label='Train')
    plt.xlabel(f'Actual ({scale})', fontsize=20, fontweight='bold')
    plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold'); plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', frameon=False, fontsize=18)
    ax = plt.gca()
    ax.text(0.99, 0.03, 'ALIGNN', transform=ax.transAxes,
            fontsize=18, fontweight='bold', color='black', va='bottom', ha='right', alpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'act_pred_res_plot_{scale}.png'), dpi=300)
    plt.close()

    # Error histogram
    plt.figure(figsize=(7, 6), dpi=300)
    errors_test = actual_test - pred_test
    sns.histplot(errors_test, kde=True, color='blue', bins=30, edgecolor='black')
    plt.xlabel(f'Prediction Error ({scale})', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold'); plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'error_histogram_kde_{scale}.png'), dpi=300)
    plt.close()

# === Main execution ===
if __name__ == "__main__":
    folder = "/home/a4724/sandeep/alignn_cal/3D/temp"
    plot_dir = os.path.join(folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    files_log = {
        "train": os.path.join(folder, "Train_results.json"),
        "val":   os.path.join(folder, "Val_results.json"),
        "test":  os.path.join(folder, "Test_results.json")
    }
    files_orig = {
        "train": os.path.join(folder, "Train_results_orig.json"),
        "val":   os.path.join(folder, "Val_results_orig.json"),
        "test":  os.path.join(folder, "Test_results_orig.json")
    }

    plot_actual_vs_pred(files_log['train'], files_log['val'], files_log['test'], plot_dir, scale='log')
    plot_actual_vs_pred(files_orig['train'], files_orig['val'], files_orig['test'], plot_dir, scale='orig')
    print(f"✅ Plots saved in {plot_dir}")

