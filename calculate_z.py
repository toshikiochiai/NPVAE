from __future__ import print_function
import torch
from torch.utils.data import DataLoader
import time
import torch.multiprocessing as mp
from rdkit import RDLogger, Chem
from collections import OrderedDict
import pickle
import datetime
import model.NPVAE as NPVAE
from train import collate as collate
import numpy as np
import warnings

def main(args):
    warnings.simplefilter('ignore')
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    #Loading
    with open(args.smiles_path) as f:
        all_smiles=[line.strip("\r\n ").split()[0] for line in f]
    with open(args.prepared_path+"/input_data/graphs","rb")as f:
        graphs=pickle.load(f)
    with open(args.prepared_path+"/input_data/labels","rb") as f:
        labels=pickle.load(f)
    with open(args.prepared_path+"/input_data/ecfp_list3D","rb")as f:
        ecfp_list3D = pickle.load(f)
    dataset=[]
    for i in range(len(graphs)):
        dataset.append((graphs[i], None, None, ecfp_list3D[i], \
                        all_smiles[i], graphs[i].ndata['ecfp'], None, \
                        None, None, None,\
                        None, None))
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    model=NPVAE.Chem_VAE(args.x_size, args.h_size, args.mid_size, args.z_dim, 0, len(labels), args.max_iter, labels, None, None, None, None, None, args.device, "test", n_trial=args.n_trial, test3D=not(args.return2D))
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

    z_list=[]
    smi_list=[]
    g_list=[]
    mol_pred_list=[]

    start=time.time()
    with torch.no_grad():
        for iter, (g, bg, bg_node, m_ecfp, smiles, feats, prop, root_answer, topo_ans_list,\
               bond_ans_list, label_ans_list, target_id_list) in enumerate(test_loader):
            g=g.to(torch.device(args.device))
            n = g.number_of_nodes()
            h = torch.zeros((n, args.h_size)).to(args.device)
            c = torch.zeros((n, args.h_size)).to(args.device)
            feat=feats[0].to(args.device)
            m_ecfp=m_ecfp[0].to(args.device)
            smiles=smiles[0]
            if args.return2D:
                z, dec_smi, g = model(g, None, None, m_ecfp, \
                                                smiles, None, feat, h, c, \
                                                None, None, None, \
                                                None, None)
                z_list.append(z.cpu().detach().numpy())
                smi_list.append(dec_smi)
                g_list.append(g)  
            else:
                z, dec_smi, g, mol_pred = model(g, None, None, m_ecfp, \
                                                smiles, None, feat, h, c, \
                                                None, None, None, \
                                                None, None)
                z_list.append(z.cpu().detach().numpy())
                smi_list.append(dec_smi)
                g_list.append(g)
                mol_pred_list.append(mol_pred.cpu().detach().numpy())
    end=time.time()
    print('Test time: {}'.format(end-start))
    #Saving
    with open(args.save_path+"/z_list","wb")as f:
        pickle.dump(z_list, f)
    with open(args.save_path+"/g_list","wb")as f:
        pickle.dump(g_list, f)
    with open(args.save_path+"/dec_smiles.txt","wt")as f:
        for v in smi_list:
            f.write(str(v)+'\n')
    if not args.return2D:
        with open(args.save_path+"/mol_pred_list","wb")as f:
            pickle.dump(mol_pred_list, f)
    print("Now Saving...")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str,
                        default="./smiles_data/drugbank_smiles.txt", help="Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument("--prepared_path", type=str,
                        default="./created_data", help="The path where the created 'input data' is saved")
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
    parser.add_argument("--save_path", type=str,
                        default="./created_data/params", help="Path to save latent variables and other output values")
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu', help="Whether the device is GPU or CPU")
    parser.add_argument("--n_trial", type=int,
                        default=3, help="Number of retries when connection fails during compound generation")
    parser.add_argument("--return2D", action='store_true', help="If this flag is set, no 3D structural information is returned")
    args = parser.parse_args()
    print("Now Calculating...")
    main(args)
    print("Finished")