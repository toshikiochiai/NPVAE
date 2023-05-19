from __future__ import print_function
from model.utils import make_ecfp2D, make_ecfp3D
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import numpy as np

def main(args):
    #Loading
    with open(args.smiles_path) as f:
        all_smiles = [line.strip("\r\n ").split()[0] for line in f]
    with open(args.saved_path + "/dec_smiles.txt") as f:
        smi_list = [line.strip("\r\n ").split()[0] for line in f]
    if args.check3D_result:
        with open(args.saved_path + "/mol_pred_list", "rb")as f:
            mol_pred_list = pickle.load(f)
    #Evaluation        
    same_smi = 0
    similar = 0
    tanimoto_list = []
    dice_list = []
    cosine_list = []
    count = 0
    Dsame = 0
    Dcount = 0
    if args.check3D_result:
        for n, s in enumerate(all_smiles):
            mol = Chem.MolFromSmiles(s)
            opts = StereoEnumerationOptions(unique = True, onlyUnassigned = True)
            isomers = list(EnumerateStereoisomers(mol, options = opts))
            if len(isomers) > 1: #For compounds that do not have a defined three-dimensional structure, we verify the consistency of the two-dimensional structure
                in_mol, in_smi = addHs_and_sanitize(s)
                dec_mol, dec_smi = addHs_and_sanitize(smi_list[n])
                in_fp = AllChem.GetMorganFingerprintAsBitVect(in_mol, 2, 2048)
                dec_fp = AllChem.GetMorganFingerprintAsBitVect(dec_mol, 2, 2048)
                tanimoto = DataStructs.TanimotoSimilarity(in_fp, dec_fp)
                dice = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.DiceSimilarity)
                cosine = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.CosineSimilarity)
                tanimoto_list.append(tanimoto)
                dice_list.append(dice)
                cosine_list.append(cosine)
                count += 1
                if in_smi == dec_smi:
                    same_smi += 1
                if tanimoto > 0.5:
                    similar += 1
            else:
                dec_mol = Chem.MolFromSmiles(smi_list[n])
                opts2 = StereoEnumerationOptions(unique = True, onlyUnassigned = False)
                dec_isomers = list(EnumerateStereoisomers(dec_mol, options = opts2))
                for i, iso in enumerate(dec_isomers):
                    iso = Chem.MolToSmiles(iso)
                    ecfp = make_ecfp3D(iso)
                    ecfp = np.array(ecfp, dtype = float)
                    d = np.linalg.norm(ecfp - mol_pred_list[n][0])
                    if i == 0:
                        min_dist = d
                        dec_smi = iso
                    elif d < min_dist:
                        min_dist = d
                        dec_smi = iso
                in_mol, in_smi = addHs_and_sanitize_3D(s)
                dec_mol, dec_smi = addHs_and_sanitize_3D(dec_smi)
                in_fp = AllChem.GetMorganFingerprintAsBitVect(in_mol, 2, 2048)
                dec_fp = AllChem.GetMorganFingerprintAsBitVect(dec_mol, 2, 2048)
                tanimoto = DataStructs.TanimotoSimilarity(in_fp, dec_fp)
                dice = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.DiceSimilarity)
                cosine = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.CosineSimilarity)
                tanimoto_list.append(tanimoto)
                dice_list.append(dice)
                cosine_list.append(cosine)
                count += 1
                Dcount += 1
                if in_smi == dec_smi:
                    same_smi += 1
                    Dsame += 1
                if tanimoto > 0.5:
                    similar += 1
        print("The same smiles: {} %".format(same_smi * 100 / count))
        print("3D prediction ACC: {} %".format(Dsame * 100 / Dcount))
        print("The similar smiles(Tanimoto Similarity>0.5): {} %".format(similar * 100 / count))
        print("[Ave] Dice: {}, Tanimoto: {}, Cosine: {}".format(np.mean(dice_list), np.mean(tanimoto_list), np.mean(cosine_list)))
    else:
        for i in range(len(smi_list)):
            in_mol, in_smi = addHs_and_sanitize(all_smiles[i])
            dec_mol, dec_smi = addHs_and_sanitize(smi_list[i])
            in_fp = AllChem.GetMorganFingerprintAsBitVect(in_mol, 2, 2048)
            dec_fp = AllChem.GetMorganFingerprintAsBitVect(dec_mol, 2, 2048)
            tanimoto = DataStructs.TanimotoSimilarity(in_fp, dec_fp)
            dice = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.DiceSimilarity)
            cosine = DataStructs.FingerprintSimilarity(in_fp, dec_fp, metric = DataStructs.CosineSimilarity)
            tanimoto_list.append(tanimoto)
            dice_list.append(dice)
            cosine_list.append(cosine)
            if in_smi == dec_smi:
                same_smi += 1
            if tanimoto > 0.5:
                similar += 1
        print("The same smiles: {} %".format(same_smi * 100 / len(smi_list)))
        print("The similar smiles(Tanimoto Similarity): {} %".format(similar * 100 / len(smi_list)))
        print("[Ave] Dice: {}, Tanimoto: {}, Cosine: {}".format(np.mean(dice_list),
                                                               np.mean(tanimoto_list),
                                                               np.mean(cosine_list)))

def addHs_and_sanitize(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    if m is None:
        raise
    smi = Chem.MolToSmiles(m, isomericSmiles = False)
    return m, smi

def addHs_and_sanitize_3D(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    if m is None:
        raise
    smi = Chem.MolToSmiles(m, isomericSmiles = True)
    return m, smi

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type = str,
                        default = "./smiles_data/drugbank_smiles.txt", help = "Path of SMILES data for input compounds")
    parser.add_argument("--saved_path", type = str,
                        default = "./output_data", help = "The path where latent variables and other output values are saved")
    parser.add_argument("-check3D", "--check3D_result", action = 'store_true', help = "Calculates the consistency of three-dimensional structures when this flag is set (Longer time required for calculations)")
    args = parser.parse_args()
    print("Now Calculating...")
    main(args)
    print("Finished")