import glob
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import spatial
import streamlit as st

import ecif
import mol2parser
from rdkit import Chem
from Bio.PDB import PDBParser
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer


# GB-Score amino acid classification.

amino_acid_classes_1 = [
    ["ARG", "LYS", "ASP", "GLU"],
    ["GLN", "ASN", "HIS", "SER", "THR", "CYS"],
    ["TRP", "TYR", "MET"],
    ["ILE", "LEU", "PHE", "VAL", "PRO", "GLY", "ALA"],
]

# RF-Score amino acids classification (All AA in one class)

amino_acid_classes_2 = [
    [
        "ARG",
        "LYS",
        "ASP",
        "GLU",
        "GLN",
        "ASN",
        "HIS",
        "SER",
        "THR",
        "CYS",
        "TRP",
        "TYR",
        "MET",
        "ILE",
        "LEU",
        "PHE",
        "VAL",
        "PRO",
        "GLY",
        "ALA",
    ]
]

# Elemental atom types for ligand and protein.

ligand_elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

protein_elements = ["H", "C", "N", "O", "S"]


# Parse a pdb file and output a dictionary contains AA classes with their elemental atom types.
# mol2 file is parsed by mol2parser (https://github.com/miladrayka/mol2parser).


def pdb_parser(pdbfile, amino_acid_classes):
    number_of_groups = len(list(amino_acid_classes))

    amino_acid_groups = zip(
        [f"group{i}" for i in range(1, number_of_groups + 1)], amino_acid_classes
    )

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    protein = parser.get_structure("", pdbfile)

    extracted_protein_atoms = {
        f"group{i}": defaultdict(list) for i in range(1, number_of_groups + 1)
    }

    for name, group in amino_acid_groups:
        for residue in protein.get_residues():
            if residue.get_resname() in group:
                for atom in residue.get_atoms():
                    extracted_protein_atoms[name][atom.element].append(list(atom.coord))

    return extracted_protein_atoms


def summation(distances, cutoff, exp):
    selected_distances = distances[distances < cutoff]
    if exp:
        feature = sum(list(map(lambda x: 1.0 / (x**exp), selected_distances)))
    else:
        feature = len(selected_distances)

    return feature


# Feature generation function for both RF-Score (feature_type="counting_sum")
# and GB-Score (exp=2, feature_type="weighting_sum").


def features_generator(
    mol2file, pdbfile, amino_acid_classes, cutoff, feature_type, exp=None
):
    ligand = mol2parser.Mol2Parser(mol2file)
    ligand.parse()

    extracted_protein_atoms = pdb_parser(pdbfile, amino_acid_classes)

    groups = extracted_protein_atoms.keys()
    feature_vector = defaultdict(list)

    for group in groups:
        for protein_element in protein_elements:
            for ligand_element in ligand_elements:
                try:
                    ligand_coords = np.array(
                        ligand.get_molecule(ligand_element, "coords"), dtype="float"
                    ).reshape(-1, 3)

                except TypeError:
                    continue

                if extracted_protein_atoms[group][protein_element]:
                    protein_coords = np.array(
                        extracted_protein_atoms[group][protein_element], dtype="float"
                    )

                    distances = spatial.distance.cdist(
                        ligand_coords, protein_coords
                    ).ravel()

                    if feature_type == "weighting_sum":
                        feature = summation(distances, cutoff, exp)
                        name = group + "_" + protein_element + "_" + ligand_element

                    elif feature_type == "counting_sum":
                        feature = summation(distances, cutoff, exp)
                        name = protein_element + "_" + ligand_element

                    feature_vector[name].append(feature)

    return feature_vector


def chembert_features(ligand_file_sdf):
    transformer_chembert = PretrainedHFTransformer(
        kind="ChemBERTa-77M-MLM", dtype=float, random_seed=42
    )

    mol = Chem.MolFromMolFile(ligand_file_sdf, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    lig_chembert_fv = transformer_chembert(mol)

    return lig_chembert_fv


def water_features(ligand_file_mol2, protein_file, name):
    subprocess.run(
        [
            "wsl",
            "./hydramap/wsitem",
            protein_file.replace("\\", "/"),
            ligand_file_mol2.replace("\\", "/"),
            "holo",
            "4.0",
            "5.0",
            "2.0",
            "10",
        ],
        shell=True,
    )
    subprocess.run(
        [
            "wsl",
            "./hydramap/getwat",
            protein_file.replace("\\", "/"),
            ligand_file_mol2.replace("\\", "/"),
            "wsite.pdb",
            "4.0",
        ],
        shell=True,
    )
    subprocess.run(["wsl", "mv", "wat.pdb", f"./Example/{name}_wat4.pdb"])
    #subprocess.run(["wsl", "echo", name, ">", "./Example/list_pdbid.txt"])
    subprocess.run(
        ["mamba", "run", "-n", "hydramap", "python", "./calFeature.py"], shell=True
    )


def suitable_rep(feature_csv, mean_std_csv):
    df = pd.read_csv(feature_csv)
    ms_df = pd.read_csv(mean_std_csv, index_col=0)

    ms_cols = set(ms_df.columns.tolist())
    cols = set(df.columns.tolist())

    add = ms_cols.difference(cols)
    sub = cols.difference(ms_cols)

    df.drop(sub, axis=1, inplace=True)
    df[list(add)] = 0.0

    df = (df - ms_df.loc["mean", :]) / ms_df.loc["std", :]

    return df


def prediction(fv, models):
    y_preds = []

    for model in models:
        xgb_reg = xgb.XGBRegressor()
        xgb_reg.load_model(model)

        fv = fv.loc[:, xgb_reg.feature_names_in_]

        y_pred = xgb_reg.predict(fv)

        y_preds.append(y_pred)

    return y_preds

with open("./Example/list_pdbid.txt", "r") as file:
    name = file.read().strip()

ligand_file_mol2 = glob.glob(".\Example\*.mol2")[0]
ligand_file_sdf = glob.glob(".\Example\*.sdf")[0]
protein_file = glob.glob(".\Example\*.pdb")[0]

dwic_fv = features_generator(
    ligand_file_mol2, protein_file, amino_acid_classes_1, 12, "weighting_sum", exp=2.0
)
oic_fv = features_generator(
    ligand_file_mol2, protein_file, amino_acid_classes_2, 12, "counting_sum", exp=None
)
ecif_fv = ecif.GetECIF(
    protein_file, ligand_file_sdf, distance_cutoff=6.0, dict_output=True
)
lig_desc_fv = ecif.GetRDKitDescriptors(ligand_file_sdf, dict_output=True)
lig_chembert_fv = chembert_features(ligand_file_sdf)
water_features(ligand_file_mol2, protein_file, name)

pd.DataFrame(dwic_fv).to_csv("./Generated_features/dwic_fv.csv", index=False)
pd.DataFrame(oic_fv).to_csv("./Generated_features/oic_fv.csv", index=False)
pd.DataFrame(ecif_fv.values(), ecif_fv.keys()).T.to_csv(
    "./Generated_features/ecif_fv.csv", index=False
)
pd.DataFrame(lig_desc_fv.values(), lig_desc_fv.keys()).T.to_csv(
    "./Generated_features/lig_desc_fv.csv", index=False
)
pd.DataFrame(lig_chembert_fv).to_csv(
    "./Generated_features/lig_chembert_fv.csv", index=False
)

csvs = [
    ("./Generated_features/oic_fv.csv", "./Files/oic_mean_std.csv"),
    ("./Generated_features/dwic_fv.csv", "./Files/dwic_mean_std.csv"),
    ("./Generated_features/ecif_fv.csv", "./Files/ecif_mean_std.csv"),
    ("./Generated_features/lig_desc_fv.csv", "./Files/rdkit_mean_std.csv"),
    ("./Generated_features/water_fv.csv", "./Files/water_mean_std.csv"),
]

feature_dfs = []

for csv1, csv2 in csvs:
    feature_dfs.append(suitable_rep(csv1, csv2))

feature_dfs.append(pd.read_csv("./Generated_features/lig_chembert_fv.csv"))

ecif_rdkit_water_rep = pd.concat(
    [feature_dfs[2], feature_dfs[3], feature_dfs[4]], axis=1
)
ecif_rdkit_water_rep = ecif_rdkit_water_rep.loc[
    :, ~ecif_rdkit_water_rep.columns.duplicated()
]
dwic_chembert_water_rep = pd.concat(
    [feature_dfs[1], feature_dfs[5], feature_dfs[4]], axis=1
)
oic_rdkit_water_rep = pd.concat(
    [feature_dfs[0], feature_dfs[3], feature_dfs[4]], axis=1
)

models1 = glob.glob(r"./Models/xgb_ECIF_RDKit_HydraMap_*.json")
models2 = glob.glob(r"./Models/xgb_GBScore_ChemBert_HydraMap_*.json")
models3 = glob.glob(r"./Models/xgb_RFScore_RDKit_HydraMap_*.json")

predicted_ba = []

for item in [
    (ecif_rdkit_water_rep, models1),
    (dwic_chembert_water_rep, models2),
    (oic_rdkit_water_rep, models3),
]:
    predicted_ba.append(prediction(item[0], item[1]))

mean = np.mean(predicted_ba)
std = np.std(predicted_ba)
confidence_region = np.exp(std) * 0.85

with open("./Example/result.txt", "w") as file:
    file.write(str(mean))
    file.write("\n")
    file.write(str(confidence_region))