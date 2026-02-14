import torch
from torch_geometric.data import Data
from rdkit import Chem

graph_list = []
failed_mols = []

for idx, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            failed_mols.append(row['smiles'])
            continue
        mol = Chem.AddHs(mol)
        atoms = mol.GetAtoms()
        n_atoms = int(row['n_atoms'])
        z = []
        pos = []
        for j in range(1, n_atoms + 1):
            x = row[f"atom{j}_x"]
            y = row[f"atom{j}_y"]
            z_pos = row[f"atom{j}_z"]
            pos.append([x, y, z_pos])
            if j-1 < len(atoms):
                z.append(atoms[j-1].GetAtomicNum())
            else:
                z.append(6)  # fallback atomic number (carbon)
        if len(z) == 0:
            continue
        z = torch.tensor(z, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y_val = row['log_electrophilicity_index']
        if torch.isnan(torch.tensor([y_val])) or torch.isinf(torch.tensor([y_val])):
            continue
        y = torch.tensor([y_val], dtype=torch.float)

        # Print check: number of atoms and coordinate dimensions
        print(f"Mol idx {idx}: n_atoms={n_atoms}, positions count={len(pos)}, total coords={3*len(pos)}")

        graph_list.append(Data(z=z, pos=pos, y=y))
    except Exception as e:
        failed_mols.append(row['smiles'])
        continue

print(f"Successfully embedded {len(graph_list)} molecules.")
print(f"Failed to embed {len(failed_mols)} molecules.")
