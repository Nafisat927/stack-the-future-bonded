from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import pandas as pd

FILTERED_PATH = "datasets/compounds_filtered.csv"
df = pd.read_csv(FILTERED_PATH, dtype={"Smiles": str})
df = df[["Name", "Smiles", "Molecular Weight", "AlogP", "HBA", "HBD"]]
compounds = df.to_dict(orient="records")


def match_compounds(molw, lip, hba, hbd):
    if molw == "n/a" and lip == "n/a" and hba == "n/a" and hbd == "n/a":
        return compounds

    res = []

    for comp in compounds:
        smiles = str(comp["Smiles"]).strip()
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        mol_weight = Descriptors.MolWt(mol)
        mol_lip = Crippen.MolLogP(mol)
        mol_hba = Lipinski.NumHAcceptors(mol)
        mol_hbd = Lipinski.NumHDonors(mol)

        if not in_range(mol_weight, molw):
            continue
        if not in_range(mol_lip, lip):
            continue
        if not in_range(mol_hba, hba):
            continue
        if not in_range(mol_hbd, hbd):
            continue

        res.append(comp)

    return res


def in_range(value, pref):
    if pref == "n/a":
        return True

    if pref.endswith("+"):
        return value >= float(pref[:-1])

    if pref.startswith("<"):
        return value < float(pref[1:])

    if "-" in pref:
        low, high = pref.split("-")
        return float(low) <= value <= float(high)

    return True


res = match_compounds("201-300", "1-3", "0-10", "0-5")
print(res)
