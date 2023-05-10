from __future__ import print_function
import pickle
import gzip
import npscorer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem
import warnings
import torch
import model.NPVAE as NPVAE
import time
from collections import OrderedDict

def main(args):
    warnings.simplefilter('ignore')
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    #Loading
    with open(args.smiles_path) as f:
        all_smiles=[line.strip("\r\n ").split()[0] for line in f]
    with open(args.saved_path+"/params/z_list","rb")as f:
        z_list=pickle.load(f)
    with open(args.prepared_path+"/input_data/labels","rb") as f:
        labels=pickle.load(f)
    try:
        idx=all_smiles.index(args.target_smiles)
    except:
        raise Exception("'target_smiles' must be included in the SMILES data in the original 'smiles_path'")
    model=NPVAE.Chem_VAE(args.x_size, args.h_size, args.mid_size, args.z_dim, 0, len(labels), args.max_iter, labels, None, None, None, None, None, args.device, "test", n_trial=args.n_trial, test3D=not(args.only2D))
    #Loading parameters
    load_path=args.load_path+'/model.iter-{}'.format(args.load_epoch)
    state_dict = torch.load(load_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name= k.replace('module.','')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model=model.to(args.device)
    #Generation
    r_list=[]
    for i in range(args.num_gen_mols):
        r=np.random.randn(1, 256)*args.search_radius
        r_list.append(torch.from_numpy(r).float())
    new_smiles=set()
    dinfo=[]
    with torch.no_grad():
        for i in range(args.num_gen_mols):
            if (i+1)%1000==0:
                print("{}/{}".format(i+1,args.num_gen_mols))
            new_z=torch.tensor(z_list[idx])+r_list[i]
            new_z=new_z.to(args.device)
            if args.only2D:
                s, _ = model.decoder(new_z, g=None, bg=None, bg_node_list=None,\
                                               m_ecfp=None, z_dim=args.z_dim, h_size=args.h_size, MAX_ITER=args.max_iter, \
                                               prop=None, root_answer=None, t_ans_list=None, \
                                               b_ans_list=None, l_ans_list=None, t_id_list=None)
                if s not in new_smiles:
                    new_smiles.add(s)              
            else:
                s, _, mol_pred = model.decoder(new_z, g=None, bg=None, bg_node_list=None,\
                                               m_ecfp=None, z_dim=args.z_dim, h_size=args.h_size, MAX_ITER=args.max_iter, \
                                               prop=None, root_answer=None, t_ans_list=None, \
                                               b_ans_list=None, l_ans_list=None, t_id_list=None)
                if s not in new_smiles:
                    new_smiles.add(s)
                    dinfo.append(mol_pred.cpu().detach().numpy())
    new_smiles=list(new_smiles)
    new_mols=[]
    for i in range(len(new_smiles)):
        new_mols.append(Chem.MolFromSmiles(new_smiles[i])) 
    all_smiles_set=set(all_smiles)
    uniq_mols=[]
    for m in new_mols:
        s=Chem.MolToSmiles(m)
        if s in all_smiles_set:
            continue
        else:
            uniq_mols.append(m)
            
    if args.only2D:
        tanimoto_dict=dict()
        target_fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(all_smiles[idx]), 2, 2048)
        for i,m in enumerate(uniq_mols):
            m_fp=AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            tanimoto=DataStructs.TanimotoSimilarity(target_fp, m_fp)
            tanimoto_dict[uniq_mols[i]]=tanimoto
    else:
        #Addition of 3D coordinates
        mols3D=[]
        for i,m in enumerate(uniq_mols):
            mol=Chem.AddHs(m)
            flg=AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=False)
            if flg==0:
                mols3D.append(mol)
            else:
                flg2=AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=True)
                if flg2==0:
                    mols3D.append(mol)
        tanimoto_dict=dict()
        target_fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(all_smiles[idx]), 2, 2048)
        for m in mols3D:
            m_fp=AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            tanimoto=DataStructs.TanimotoSimilarity(target_fp, m_fp)
            tanimoto_dict[m]=tanimoto
    tanimoto_dict2=sorted(tanimoto_dict.items(), key=lambda x:x[1], reverse=True)     
    new_supmols=[]
    cou=0
    for i,d in enumerate(tanimoto_dict2):
        if d[1]==1.0:
            cou+=1
            continue
        if i==args.num_new_mols+cou:
            break
        else:
            new_supmols.append(d[0])
    #Writing to SDF file
    sdf_path=args.saved_path+"/new_molecules.sdf"
    writer = Chem.SDWriter(sdf_path)
    maxlen=len(str(args.num_new_mols))
    for i,m in enumerate(new_supmols):
        name="D"+str(i).zfill(maxlen)
        m.SetProp("_Name",name)
        writer.write(m)
    writer.close()
        
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str,
                        default="./smiles_data/drugbank_smiles.txt", help="Path of SMILES data for input compounds")
    parser.add_argument("--prepared_path", type=str,
                        default="./created_data", help="The path where the created 'input data' is saved")
    parser.add_argument("--saved_path", type=str,
                        default="./created_data", help="The path where the created 'parmas'(latent variables) is saved")
    parser.add_argument("-target", "--target_smiles", type=str, required=True, help="SMILES of the target point compound") 
    parser.add_argument("-ngen", "--num_gen_mols", type=int,
                        default=10000, help="Number of new compounds generated before refining") 
    parser.add_argument("-nmol", "--num_new_mols", type=int,
                        default=5000, help="Number of new compounds generated after refining (must be smaller than 'num_gen_mols')") 
    parser.add_argument("-r", "--search_radius", type=float,
                        default=1.0, help="Search radius from the target compound")
    parser.add_argument("--only2D", action='store_true', help="If this flag is set, no 3D structural information is calculated")
    parser.add_argument("--x_size", type=int,
                        default=2048, help="Dimension of the input (ECFP)")
    parser.add_argument("--h_size", type=int,
                        default=512, help="Dimension of the hidden layer")
    parser.add_argument("--mid_size", type=int,
                        default=256, help="Dimension of the middle layer")
    parser.add_argument("--z_dim", type=int,
                        default=256, help="Dimension of the latent variable")
    parser.add_argument("--max_iter", type=int,
                        default=500, help="Maximum number of iterations in the decoding process (should be set to a large value when dealing with huge molecules)")
    parser.add_argument("--load_epoch", type=int,
                        default=100, help="Epoch specified for loading parameters")
    parser.add_argument("--load_path", type=str,
                        default="./created_data/params", help="The path where learned parameters are saved")
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu', help="Whether the device is GPU or CPU")
    parser.add_argument("--n_trial", type=int,
                        default=3, help="Number of retries when connection fails during compound generation") 
    args = parser.parse_args()
    print("Now Calculating...")
    main(args)
    print("Finished")