from __future__ import print_function
from model.utils import *
from rdkit import Chem
from collections import Counter
import numpy as np
import torch
import pickle

def main(args):
    with open(args.smiles_path) as f:
        all_smiles=[line.strip("\r\n ").split()[0] for line in f]
    print("Number of SMILES entered: ",len(all_smiles))
    print("Preprocessing consists of nine processes")
    
    print("Process 1/9 is running", end='...')
    mols=[]
    cou=0
    for i in range(len(all_smiles)):
        try:
            mol=Chem.MolFromSmiles(all_smiles[i])
            mol=sanitize(mol,kekulize=False)
            mol=Chem.RemoveHs(mol)
            mols.append(mol)
        except:
            cou+=1
    if cou>0:
        raise ValueError("There might be some errors. Check your SMILES data.")
    print('done')
    
    print("Process 2/9 is running", end='...')
    count_labels=[] #(substructureSMILES,(AtomIdx in substructure, join order)xN)->frequency of use of label
    fragments=[]
    for i,m in enumerate(mols):
        cl, frag = count_fragments(m)
        count_labels += cl
        fragments.append(frag)
    count_labels=Counter(count_labels)
    print('done')
    
    print("Process 3/9 is running", end='...')
    mapidx_list=[]
    labelmap_dict_list=[]
    bondtype_list=[]
    max_mapnum_list=[]
    fragments_list=[]
    fragments=[]
    for i,m in enumerate(mols):
        mapidxs, labelmap_dict, bondtypes, max_mapnums, frag = find_fragments(m, count_labels, args.frequency)
        mapidx_list.append(mapidxs)
        labelmap_dict_list.append(labelmap_dict)
        bondtype_list.append(bondtypes)
        max_mapnum_list.append(max_mapnums)
    print('done')
        
    print("Process 4/9 is running", end='...')
    rev_labelmap_dict_list=[]
    for lm_d in labelmap_dict_list:
        rev_labelmap_dict, deg=revise_maps(lm_d)
        rev_labelmap_dict_list.append(rev_labelmap_dict)
    print('done')
    
    print("Process 5/9 is running", end='...')
    labels=[]
    for ld in rev_labelmap_dict_list:
        for k in ld.keys():
            for l in ld[k]:
                label=[]
                label.append(k)
                label.append(l)
                if label not in labels:
                    labels.append(label)
    print('done')
    
    print("Process 6/9 is running", end='...')
    graphs=[]
    sub_trees=[]
    root_ans_list=[]
    label_ans_list=[]
    bond_ans_list=[]
    for i in range(len(mapidx_list)):
        g, sub_tree, root_answer, l_ans_list, b_ans_list=make_graph(mapidx_list[i], labelmap_dict_list[i], rev_labelmap_dict_list[i] ,labels, bondtype_list[i])
        graphs.append(g)
        sub_trees.append(sub_tree)
        root_ans_list.append(root_answer)
        label_ans_list.append(l_ans_list)
        bond_ans_list.append(b_ans_list)
    print('done')
    
    print("Process 7/9 is running", end='...')
    l_1_counter=np.zeros(len(labels),dtype=int)
    l_counter=np.zeros(len(labels),dtype=int)
    b_counter=np.zeros(3,dtype=int)
    t_counter=np.zeros(2,dtype=int)
    bg_node_list=[]
    target_id_list=[]
    topo_ans_list=[]
    for i in range(len(graphs)):
        _, bg_node_l, target_id_l, topo_ans_l, \
        l_1_counter, l_counter, b_counter, t_counter = \
        demon_decoder(graphs[i], sub_trees[i], \
                      root_ans_list[i], label_ans_list[i], bond_ans_list[i], \
                      l_1_counter, l_counter, b_counter, t_counter, \
                      labels)
        bg_node_list.append(bg_node_l)
        target_id_list.append(target_id_l)
        topo_ans_list.append(topo_ans_l)
    print('done')
    
    print("Process 8/9 is running", end='...')
    #Creating weights for cross-entropy
    t_wei=[]
    b_wei=[]
    l_wei=[]
    l_1_wei=[]
    t_max=np.max(t_counter)
    b_max=np.max(b_counter)
    l_max=np.max(l_counter)
    l_1_max=np.max(l_1_counter)
    for i in range(len(t_counter)):
        t_wei.append(t_max/(t_counter[i]+1e-7))
    for i in range(len(b_counter)):
        b_wei.append(b_max/(b_counter[i]+1e-7))
    for i in range(len(l_counter)):
        l_wei.append(l_max/(l_counter[i]+1e-7))
        l_1_wei.append(l_1_max/(l_1_counter[i]+1e-7))
    t_wei=np.array(t_wei)
    b_wei=np.array(b_wei)
    l_wei=np.array(l_wei)
    l_1_wei=np.array(l_1_wei)
    t_weights=torch.from_numpy(t_wei).float()
    b_weights=torch.from_numpy(b_wei).float()
    l_weights=torch.from_numpy(l_wei).float()
    l_1_weights=torch.from_numpy(l_1_wei).float()
    print('done')
    
    print("Process 9/9 is running", end='...')
    ecfp_list3D=[]
    for i in range(len(all_smiles)):
        mol_ecfp=make_ecfp3D(all_smiles[i], args.fpbit, args.radius)
        mol_ecfp=np.asarray(mol_ecfp)
        mol_ecfp=torch.from_numpy(mol_ecfp).float()
        mol_ecfp=mol_ecfp.unsqueeze(0)
        ecfp_list3D.append(mol_ecfp)
    m_counter=np.zeros(args.fpbit, dtype=int)
    for fp in ecfp_list3D:
        for i,bit in enumerate(fp[0]):
            if bit:
                m_counter[i]+=1
    m_wei=[]
    m_max=np.max(m_counter)
    for i in range(len((m_counter))):
        m_wei.append(m_max/(m_counter[i]+1e-7))
    m_wei=np.array(m_wei)
    m_weights=torch.from_numpy(m_wei).float()
    print('done')
    
    print("Saving the created data to the specified path", end='...')
    #Writing the created lists
    with open(args.save_path+"/input_data/graphs","wb")as f:
        pickle.dump(graphs,f)
    with open(args.save_path+"/input_data/sub_trees","wb")as f:
        pickle.dump(sub_trees,f)
    with open(args.save_path+'/input_data/labels',"wb") as f:
        pickle.dump(labels,f)
    with open(args.save_path+"/input_data/weights/t_weights","wb")as f:
        pickle.dump(t_weights,f)
    with open(args.save_path+"/input_data/weights/b_weights","wb")as f:
        pickle.dump(b_weights,f)
    with open(args.save_path+"/input_data/weights/l_weights","wb")as f:
        pickle.dump(l_weights,f)
    with open(args.save_path+"/input_data/weights/l_1_weights","wb")as f:
        pickle.dump(l_1_weights,f)
    with open(args.save_path+"/input_data/ecfp_list3D","wb")as f:
        pickle.dump(ecfp_list3D,f)
    with open(args.save_path+"/input_data/weights/m_weights","wb")as f:
        pickle.dump(m_weights,f)
    with open(args.save_path+"/input_data/bg_node_list","wb")as f:
        pickle.dump(bg_node_list,f)
    with open(args.save_path+"/input_data/root_ans_list","wb")as f:
        pickle.dump(root_ans_list,f)
    with open(args.save_path+"/input_data/label_ans_list","wb")as f:
        pickle.dump(label_ans_list,f)
    with open(args.save_path+"/input_data/bond_ans_list","wb")as f:
        pickle.dump(bond_ans_list,f)
    with open(args.save_path+"/input_data/topo_ans_list","wb")as f:
        pickle.dump(topo_ans_list,f)
    with open(args.save_path+"/input_data/target_id_list","wb")as f:
        pickle.dump(target_id_list,f)
    print('done')

if __name__=="__main__":
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str,
                        default="./smiles_data/drugbank_smiles.txt", help="Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument("-freq", "--frequency", type=int, default=5,
                        help="Threshold frequencies at decomposition")
    parser.add_argument("-fpbit", type=int, default=2048,
                        help="Number of bits of ECFP")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="Effective radius of ECFP")
    parser.add_argument("--save_path", type=str,
                        default="./created_data", help="Path to save created data")
    args = parser.parse_args()
    main(args)