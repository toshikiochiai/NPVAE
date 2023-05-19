from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import copy
import numpy as np
import dgl
import torch

def set_atommap(mol, num = 0):
    for i,atom in enumerate(mol.GetAtoms(), start = num):
        atom.SetAtomMapNum(i)
    return mol

#smiles->Mol
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

#Mol->smiles
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles = True)

#Mol->Mol (Error->None)
def sanitize(mol, kekulize = True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def is_aromatic_ring(mol):
    if mol.GetNumAtoms() == mol.GetNumBonds(): 
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False
    
def copy_atom(atom, atommap = True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles = False)
    new_mol = Chem.MolFromSmiles(smiles, sanitize = False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

#Valence adjustment by hydrogen addition after decomposition
def add_Hs(rwmol, a1, a2, bond):
    if str(bond.GetBondType()) == 'SINGLE':
        num = 1
    elif str(bond.GetBondType()) == 'DOUBLE':
        num = 2
    elif str(bond.GetBondType()) == 'TRIPLE':
        num = 3
    elif str(bond.GetBondType()) == 'AROMATIC':
        print("error in add_Hs 1")
    else:
        print("error in add_Hs 2")
        
    for i in range(num):
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, a1.GetIdx(), Chem.BondType.SINGLE)
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, a2.GetIdx(), Chem.BondType.SINGLE)
    return rwmol

#Valence adjustment by removing hydrogen after connecting
def remove_Hs(rwmol, a1, a2, bond):
    try:
        if str(bond.GetBondType()) == 'SINGLE':
            num = 1
        elif str(bond.GetBondType()) == 'DOUBLE':
            num = 2
        elif str(bond.GetBondType()) == 'TRIPLE':
            num = 3
        elif str(bond.GetBondType()) == 'AROMATIC':
            print("error in remove_Hs 1")
        else:
            print("error in remove_Hs 2")
    except:
        if bond == 0:
            num = 1
        elif bond == 1:
            num = 2
        elif bond == 2:
            num = 3
        else:
            raise
    rwmol = Chem.AddHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    #Set hydrogen maps for connected atoms
    h_map1 = 2000000
    h_map2 = 3000000
    f_h_map1 = copy.copy(h_map1)
    f_h_map2 = copy.copy(h_map2)
    for b in rwmol.GetBonds():
        s_atom = b.GetBeginAtom()
        e_atom = b.GetEndAtom()
        if (e_atom.GetIdx() == a1.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (s_atom.GetIdx() == a1.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (e_atom.GetIdx() == a2.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
        elif (s_atom.GetIdx() == a2.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
    for i in range(num):
        try:
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map1 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map2 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
        except:
            print("Remove Hs times Error!!")
            raise
    rwmol = rwmol.GetMol()
    rwmol = sanitize(rwmol, kekulize = False)
    rwmol = Chem.RemoveHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    return rwmol

#Calculate frequency after decomposition
def count_fragments(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    sep_sets = [] #Set of atom maps of joints
    set_idx = 10000 #Temporarily allocate a large Map
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        #If both are inside the ring, split there.
        if a1.IsInRing() and a2.IsInRing():
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))
        #If one atom is in a ring and the other has a bond order greater than 2, split there.
        elif (a1.IsInRing() and a2.GetDegree() > 1) or (a2.IsInRing() and a1.GetDegree() > 1):
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))   
    sep_idx = 1
    atommap_dict = defaultdict(list) #key->AtomIdx, value->sep_idx (In the whole compound before decomposition)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if ((a1.GetIdx(),a2.GetIdx()) in sep_sets) or ((a2.GetIdx(),a1.GetIdx()) in sep_sets):
            a1map = new_mol.GetAtomWithIdx(a1.GetIdx()).GetAtomMapNum()
            a2map = new_mol.GetAtomWithIdx(a2.GetIdx()).GetAtomMapNum()
            atommap_dict[a1map].append(sep_idx)
            atommap_dict[a2map].append(sep_idx)
            new_mol = add_Hs(new_mol, a1, a2, bond)
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            sep_idx += 1
    for i in range(len(atommap_dict)):
        atommap_dict[i] = sorted(atommap_dict[i])  
    for i in list(atommap_dict.keys()):
        if atommap_dict[i] == []:
            atommap_dict.pop(i)
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragments = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragments = [sanitize(fragment, kekulize = False) for fragment in fragments]
    count_labels = []
    for i, fragment in enumerate(fragments):
        order_list = [] #Stores join orders in the substructures
        count_label = []
        frag_mol = copy.deepcopy(fragment)
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
        frag_smi = Chem.MolToSmiles(sanitize(frag_mol, kekulize = False))
        #Fix AtomIdx as order changes when AtomMap is deleted.
        atom_order = list(map(int, frag_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                order_list.append(atommap_dict[amap])
        order_list = sorted(order_list)
        count_label.append(frag_smi)
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                count_label.append(atom_order.index(atom.GetIdx()))
                count_label.append(order_list.index(atommap_dict[amap]) + 1)
        count_labels.append(tuple(count_label))
    return count_labels, fragments

#Create a decomposed list
def find_fragments(mol, count_labels, count_thres):
    mol = Chem.rdmolops.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    new_mol = Chem.RWMol(mol)
    new_mol2 = copy.deepcopy(new_mol)
    
    sep_sets = [] 
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsInRing() and a2.IsInRing():
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))
        elif (a1.IsInRing() and a2.GetDegree() > 1) or (a2.IsInRing() and a1.GetDegree() > 1):
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))
        
    sep_idx = 1
    atommap_dict = defaultdict(list)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if ((a1.GetIdx(),a2.GetIdx()) in sep_sets) or ((a2.GetIdx(),a1.GetIdx()) in sep_sets):
            a1map = new_mol.GetAtomWithIdx(a1.GetIdx()).GetAtomMapNum()
            a2map = new_mol.GetAtomWithIdx(a2.GetIdx()).GetAtomMapNum()
            atommap_dict[a1map].append(sep_idx)
            atommap_dict[a2map].append(sep_idx)
            new_mol = add_Hs(new_mol, a1, a2, bond)
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            sep_idx += 1
    for i in range(len(atommap_dict)):
        atommap_dict[i] = sorted(atommap_dict[i])  
    for i in list(atommap_dict.keys()):
        if atommap_dict[i] == []:
            atommap_dict.pop(i)
    
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragments = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragments = [sanitize(fragment, kekulize = False) for fragment in fragments]
        
    for i, fragment in enumerate(fragments):
        have_ring = False
        order_list = []
        count_label = []
        frag_mol = copy.deepcopy(fragment)
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
        frag_smi = Chem.MolToSmiles(sanitize(frag_mol, kekulize = False))
        for atom in fragment.GetAtoms():
            if atom.IsInRing():
                have_ring = True 
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                order_list.append(atommap_dict[amap])
        order_list = sorted(order_list)
        count_label.append(frag_smi)
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                count_label.append(atom.GetIdx())
                count_label.append(order_list.index(atommap_dict[amap]) + 1)
        count = count_labels[tuple(count_label)]
        if count < count_thres and have_ring == False:
            set_idx=10000
            #Query for substructure search
            query = Chem.MolFromSmiles('C(=O)N')
            m_list = list(new_mol2.GetSubstructMatches(query))
            q_list = [] #Index of query match
            for i in range(len(m_list)):
                for j in range(len(m_list[i])):
                    if m_list[i][j] not in q_list:
                        q_list.append(m_list[i][j])

            query2 = Chem.MolFromSmiles('C(=O)O')
            m_list2 = list(new_mol2.GetSubstructMatches(query2))
            q_list2 = [] #Index of query match
            for i in range(len(m_list2)):
                for j in range(len(m_list2[i])):
                    if m_list2[i][j] not in q_list2:
                        q_list2.append(m_list2[i][j])

            query3 = Chem.MolFromSmiles('C(=O)')
            m_list3 = list(new_mol2.GetSubstructMatches(query3))
            q_list3 = [] #Index of query match
            for i in range(len(m_list3)):
                for j in range(len(m_list3[i])):
                    if m_list3[i][j] not in q_list3:
                        q_list3.append(m_list3[i][j])

            query4 = Chem.MolFromSmiles('CO')
            m_list4 = list(new_mol2.GetSubstructMatches(query4))
            q_list4 = [] #Index of query match
            for i in range(len(m_list4)):
                for j in range(len(m_list4[i])):
                    if m_list4[i][j] not in q_list4:
                        q_list4.append(m_list4[i][j])

            for bond in fragment.GetBonds():
                if bond.IsInRing(): continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ###Amide bond or amide group###
                #C side
                if ((a1.GetAtomMapNum() in q_list) and (a1.GetSymbol() == 'C') and (a1.GetDegree() == 3)) \
                and (a2.GetSymbol() != 'H')and(a2.GetAtomMapNum() not in q_list):
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum() + set_idx)
                elif ((a2.GetAtomMapNum() in q_list) and (a2.GetSymbol() == 'C') and (a2.GetDegree() == 3)) \
                and (a1.GetSymbol() != 'H')and(a1.GetAtomMapNum() not in q_list):
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum()+set_idx)
                #N side
                elif ((a1.GetAtomMapNum() in q_list) and (a1.GetSymbol() == 'N') and (a1.GetDegree() == 2)) \
                and (a2.GetSymbol() != 'H')and(a2.GetAtomMapNum() not in q_list):
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum()+set_idx)
                elif ((a2.GetAtomMapNum() in q_list) and (a2.GetSymbol() == 'N') and (a2.GetDegree() == 2)) \
                and (a1.GetSymbol() != 'H')and(a1.GetAtomMapNum() not in q_list):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum() + set_idx)

            for bond in fragment.GetBonds():
                if bond.IsInRing(): continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ###Ester bond or carboxy group###
                #C side
                if ((a1.GetAtomMapNum() in q_list2) and (a1.GetSymbol() == 'C') and (a1.GetDegree() == 3)) \
                and (a2.GetSymbol() != 'H')and(a2.GetAtomMapNum() not in q_list2):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum() + set_idx)
                elif ((a2.GetAtomMapNum() in q_list2) and (a2.GetSymbol() == 'C') and (a2.GetDegree() == 3)) \
                and (a1.GetSymbol() != 'H')and(a1.GetAtomMapNum() not in q_list2):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum() + set_idx)
                #O side
                elif ((a1.GetAtomMapNum() in q_list2) and (a1.GetSymbol() == 'O') and (a1.GetDegree() == 2)) \
                and (a2.GetSymbol() != 'H')and(a2.GetAtomMapNum() not in q_list2):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >=  set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum() + set_idx)
                elif ((a2.GetAtomMapNum() in q_list2) and (a2.GetSymbol() == 'O') and (a2.GetDegree() == 2)) \
                and (a1.GetSymbol() != 'H')and(a1.GetAtomMapNum() not in q_list2):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))
                    fragment.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum() + set_idx)

            for bond in fragment.GetBonds():
                if bond.IsInRing(): continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ###Ketone group or Aldehyde group###
                if (((a1.GetAtomMapNum() in q_list3) and (a1.GetSymbol() == 'C') and (a1.GetDegree() > 1)) \
                and (a2.GetSymbol() != 'H')and(a2.GetSymbol() != 'N')and(a2.GetSymbol() != 'O')and(a2.GetAtomMapNum() not in q_list3)) \
                or (((a2.GetAtomMapNum() in q_list3) and (a2.GetSymbol() == 'C') and (a2.GetDegree() > 1)) \
                and (a1.GetSymbol() != 'H')and(a1.GetSymbol() != 'N')and(a1.GetSymbol() != 'O')and(a1.GetAtomMapNum() not in q_list3)):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))

            for bond in fragment.GetBonds():
                if bond.IsInRing(): continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ###Ether bond or hydroxy group###
                if (((a1.GetAtomMapNum() in q_list4) and (a1.GetSymbol() == 'C')) \
                and (a2.GetSymbol() == 'O')) \
                or (((a2.GetAtomMapNum() in q_list4) and (a2.GetSymbol() == 'C')) \
                and (a1.GetSymbol() == 'O')):
                    #If it's already decomposed by a higher priority functional group, then nothing.
                    if (a1.GetAtomMapNum() >= set_idx or a2.GetAtomMapNum() >= set_idx):
                        continue
                    sep_sets.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))

    sep_idx = 1
    bondtype_list = []
    atommap_dict = defaultdict(list)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if ((a1.GetIdx(),a2.GetIdx()) in sep_sets) or ((a2.GetIdx(),a1.GetIdx()) in sep_sets):
            a1map = new_mol2.GetAtomWithIdx(a1.GetIdx()).GetAtomMapNum()
            a2map = new_mol2.GetAtomWithIdx(a2.GetIdx()).GetAtomMapNum()
            atommap_dict[a1map].append(sep_idx)
            atommap_dict[a2map].append(sep_idx)
            bondtype_list.append(str(bond.GetBondType()))
            new_mol2 = add_Hs(new_mol2, a1, a2, bond)
            new_mol2.RemoveBond(a1.GetIdx(), a2.GetIdx())
            sep_idx += 1
    for i in range(len(atommap_dict)):
        atommap_dict[i] = sorted(atommap_dict[i])
    for i in list(atommap_dict.keys()):
        if atommap_dict[i] == []:
            atommap_dict.pop(i)
    max_mapnum = sep_idx - 1
    new_mol2 = new_mol2.GetMol()
    new_mol2 = sanitize(new_mol2, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol2)
    fragments = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragments = [sanitize(fragment, kekulize = False) for fragment in fragments]

    mapidx_list = [] #Set:(substructureSMILES, junctionAtomIdx) for graph and adjacency matrix creation
    labelmap_dict = defaultdict(list) #key->label_smiles, value->fragmap_dict
    for i, fragment in enumerate(fragments):
        fragmap_dict = defaultdict(list) #key->AtomIdx, value->sep_idx(In each compound after decomposition)
        fragmap_dict2 = defaultdict(list)
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                fragmap_dict[atom.GetIdx()].append(atommap_dict[amap])
            fragment.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
        frag_smi = Chem.MolToSmiles(fragment)
        atom_order = list(map(int, fragment.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
        for fragmap_v in list(fragmap_dict.keys()):
            val = fragmap_dict.pop(fragmap_v)
            fragmap_dict2[atom_order.index(fragmap_v)] = val
        fragmap_dict = fragmap_dict2
        for j in range(len(fragmap_dict)):
            fragmap_dict[j] = sorted(fragmap_dict[j])   
        if frag_smi in labelmap_dict.keys():
            if labelmap_dict[frag_smi] not in list(fragmap_dict.values()):
                labelmap_dict[frag_smi].append(fragmap_dict)
        else:
            labelmap_dict[frag_smi].append(fragmap_dict)
        midx = labelmap_dict[frag_smi].index(fragmap_dict)
        mapidx_list.append((frag_smi, midx))
    for values in labelmap_dict.values():
        for v in values:
            for i in list(v.keys()):
                if v[i] == []:
                    v.pop(i)
                else:
                    v[i] = v[i][0]
    return mapidx_list, labelmap_dict, bondtype_list, max_mapnum, fragments

def revise_maps(labelmap_dict):
    rev_labelmap_dict = copy.deepcopy(labelmap_dict)
    max_deg = 0
    for values in rev_labelmap_dict.values():
        for v in values:
            maplist = []
            for i in list(v.keys()):
                maplist += v[i]
                if len(v[i]) > max_deg:
                    max_deg = len(v[i])
            maplist = sorted(maplist)
            for i in list(v.keys()):
                for j in range(len(v[i])):
                    v[i][j] = maplist.index(v[i][j]) + 1
    return rev_labelmap_dict, max_deg

def make_ecfp2D(smiles, n_bit = 2048, r = 2):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, r, n_bit, useChirality = False)
    return ecfp

def make_ecfp3D(smiles, n_bit = 2048, r = 2):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, r, n_bit, useChirality = True)
    return ecfp

#Partial tree creation with unnecessary nodes removed
def make_subtree(tree):
    flag = 1
    while(flag == 1):
        flag = 0
        for node in range(tree.number_of_nodes()):
            deg = tree.in_degrees(node) + tree.out_degrees(node)
            if deg == 0:
                tree = dgl.remove_nodes(tree,node)
                flag = 1
                break
    return tree

def set_bondlabel(bondtype):
    if bondtype == 'SINGLE':
        b_label = torch.tensor([0])
    elif bondtype == 'DOUBLE':
        b_label = torch.tensor([1])
    elif bondtype == 'TRIPLE':
        b_label = torch.tensor([2])
    else:
        raise
    return b_label

#Creating Graphs
def make_graph(mapidx_list, labelmap_dict, rev_labelmap_dict, labels, bondtype_list):
    map_dict = defaultdict(list) #Stores which part (key) has which Index (value)
    mg = dgl.DGLGraph()
    sub_tree = []
    l_ans_list = []
    b_ans_list = []
    for i, (smi, fragidx) in enumerate(mapidx_list):
        for l in labelmap_dict[smi][fragidx].values():
            for idx in l:
                map_dict[i].append(idx)
    if len(map_dict) == 0:
        mg.add_nodes(1)
        fp = make_ecfp2D(mapidx_list[0][0])
        feat = torch.from_numpy(np.array(fp)).float()
        feat = feat.unsqueeze(0)
        mg.ndata['ecfp'] = feat
        sub_tree.append(mg)
        label = []
        label.append(mapidx_list[0][0])
        label.append(rev_labelmap_dict[mapidx_list[0][0]][mapidx_list[0][1]])
        root_answer = torch.tensor([labels.index(label)])
        return mg, sub_tree, root_answer, l_ans_list, b_ans_list
    else:
        max_idx = 0
        for l in map_dict.values():
            for v in l:
                if v > max_idx:
                    max_idx = v
        cidx = 1 #map to connect
        pair_idx = []
        track = dict() # key: get index in part, value: node number in graph
        nid = 0
        for n in range(i + 1):
            if cidx in map_dict[n]:
                pair_idx.append(n)
                track[n] = nid
                nid += 1
                if len(pair_idx) == 2:
                    break
        if max(map_dict[pair_idx[1]]) > max(map_dict[pair_idx[0]]):
            pair_idx[0], pair_idx[1] = pair_idx[1], pair_idx[0]
            track[pair_idx[0]], track[pair_idx[1]] = track[pair_idx[1]], track[pair_idx[0]]
        mg.add_nodes(1)
        fp = np.array(make_ecfp2D(mapidx_list[pair_idx[0]][0]))
        feat1 = torch.from_numpy(np.array(fp)).float()
        feat1 = feat1.unsqueeze(0)
        mg.ndata['ecfp'] = feat1
        sub_tree.append(copy.deepcopy(mg))
        label = []
        label.append(mapidx_list[pair_idx[0]][0])
        label.append(rev_labelmap_dict[mapidx_list[pair_idx[0]][0]][mapidx_list[pair_idx[0]][1]])
        root_answer = torch.tensor([labels.index(label)])
        mg.add_nodes(1)
        mg.add_edges(0,1)
        fp = np.array(make_ecfp2D(mapidx_list[pair_idx[1]][0]))
        feat2 = torch.from_numpy(np.array(fp)).float()
        feat2 = feat2.unsqueeze(0)
        feat = torch.cat((feat1, feat2), 0)
        mg.ndata['ecfp'] = feat
        sub_tree.append(copy.deepcopy(mg))
        label = []
        label.append(mapidx_list[pair_idx[1]][0])
        label.append(rev_labelmap_dict[mapidx_list[pair_idx[1]][0]][mapidx_list[pair_idx[1]][1]])
        l_ans_list.append(torch.tensor([labels.index(label)]))
        b_ans_list.append(set_bondlabel(bondtype_list[0]))
        if max_idx > 1:
            for cidx in range(2, max_idx + 1):
                pairs = []
                pair_idx = []
                for n in range(i + 1):
                    if cidx in map_dict[n]:
                        pairs.append(Chem.MolFromSmiles(mapidx_list[n][0]))
                        pair_idx.append(n)
                        if n not in list(track.keys()):
                            new_idx = n
                            track[n] = cidx
                if len(pair_idx) != 2:
                    raise
                mg.add_nodes(1)
                if mg.in_degrees(track[pair_idx[1]]) + mg.out_degrees(track[pair_idx[1]]) == 0:
                    mg.add_edges(track[pair_idx[0]], track[pair_idx[1]])
                else:
                    mg.add_edges(track[pair_idx[1]], track[pair_idx[0]])
                fp_n = np.array(make_ecfp2D(mapidx_list[new_idx][0]))
                feat_n = torch.from_numpy(np.array(fp_n)).float()
                feat_n = feat_n.unsqueeze(0)
                feat = torch.cat((feat, feat_n), 0)
                mg.ndata['ecfp'] = feat
                sub_tree.append(copy.deepcopy(mg))
                label = []
                label.append(mapidx_list[new_idx][0])
                label.append(rev_labelmap_dict[mapidx_list[new_idx][0]][mapidx_list[new_idx][1]])
                l_ans_list.append(torch.tensor([labels.index(label)]))
                b_ans_list.append(set_bondlabel(bondtype_list[cidx - 1]))
        mg = make_subtree(mg)
        assert mg.number_of_nodes() == len(mapidx_list)
        return mg, sub_tree, root_answer, l_ans_list, b_ans_list
    
def demon_decoder(g, sub_tree, root_ans, label_ans_l, bond_ans_l, \
                  l_1_counter, l_counter, b_counter, t_counter, labels, MAX_ITER = 500):
    target_id_l = []
    numnd = 0
    kaisa = 1
    bg_node_l = [] #Tuple of (node ID when graph batching, backtrack or not)
    topo_ans_l = []
    numatom = 0
    numbond = 0
    track = []
    map_track = []
    ITER = 0
    while(ITER < (MAX_ITER + 1)):
        if ITER == 0:
            label_ans = root_ans
            l_1_counter[label_ans] += 1
            target_id = 0
            numatom += 1
            dec_smi = labels[label_ans][0]
            dec_mol = setmap_to_mol(Chem.MolFromSmiles(dec_smi), target_id)
            track.append(target_id)
            target_id_l.append(target_id)
            map_track.append(labels[label_ans][1])
            bg_node_l.append((numnd,0))
            numnd += kaisa
            kaisa += 1
                    
        elif ITER > 0:
            if g.out_degrees(target_id) - (track.count(target_id) - 1) == 0:
                topo_ans = 1
                topo_ans_l.append(torch.tensor([1]))
            else:
                topo_ans = 0
                topo_ans_l.append(torch.tensor([0]))
            t_counter[topo_ans] += 1
            if topo_ans == 1: #STOP->Backtrack
                if ITER == 1:
                    break
                else:
                    try:
                        target_id = tree.predecessors(target_id).cpu()
                        target_id = int(target_id)
                        track.append(target_id)
                        target_id_l.append(target_id)
                        map_track.pop(-1)
                        bg_node_l.append((numnd + target_id - kaisa + 1, 1))  
                    except: #no parents
                        break
                    
            elif topo_ans == 0: #Create a child_node
                tree = sub_tree[numatom]
                #Bond Prediction
                bond_ans = bond_ans_l[numbond]
                b_counter[bond_ans] += 1
                #label Prediction
                new_target_id = numatom
                label_ans = label_ans_l[new_target_id - 1]
                l_counter[label_ans] += 1
                #Connect
                suc_smi = labels[label_ans][0]
                suc_mol = setmap_to_mol(Chem.MolFromSmiles(suc_smi), new_target_id)
                if target_id == 0:
                    for amap in map_track[-1].keys():
                        if track.count(target_id) in map_track[-1][amap]:
                            dec_conidx = 1000 * target_id + amap
                else:
                    for amap in map_track[-1].keys():
                        if track.count(target_id) + 1 in map_track[-1][amap]:
                            dec_conidx = 1000 * target_id + amap
                for amap in labels[label_ans][1].keys():
                    if 1 in labels[label_ans][1][amap]:
                        suc_conidx = 1000 * new_target_id + amap
                dec_mol, Connecting = connect_smiles(dec_mol, dec_conidx, suc_mol, suc_conidx, bond_ans)
                if Connecting == 0:
                    raise
                target_id = new_target_id
                numbond += 1
                numatom += 1
                track.append(target_id)
                target_id_l.append(target_id)
                map_track.append(labels[label_ans][1])    
                bg_node_l.append((numnd + target_id,0))
                numnd += kaisa
                kaisa += 1
        ITER += 1
    for atom in dec_mol.GetAtoms():
        dec_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0) 
    dec_smi = Chem.MolToSmiles(sanitize(dec_mol, kekulize = False))
    return dec_smi, bg_node_l, target_id_l, topo_ans_l,\
l_1_counter, l_counter, b_counter, t_counter 

#Add AtomMap to substructure corresponding to NodeID
def setmap_to_mol(mol, node_id):
    for atom in mol.GetAtoms():
        mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(node_id * 1000 + atom.GetIdx())
    return mol

def connect_smiles(dec_mol, dec_conidx, suc_mol, suc_conidx, bond_label):
    if bond_label == 0:
        bond_type = Chem.BondType.SINGLE
    elif bond_label == 1:
        bond_type = Chem.BondType.DOUBLE
    elif bond_label == 2:
        bond_type = Chem.BondType.TRIPLE
    else:
        raise
    con_smi = Chem.MolToSmiles(dec_mol) + '.' + Chem.MolToSmiles(suc_mol)
    con_mol = Chem.MolFromSmiles(con_smi)
    rw_mol = Chem.RWMol(con_mol)
    con_atom = []
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomMapNum() == dec_conidx or atom.GetAtomMapNum() == suc_conidx:
            con_atom.append(atom)
    if len(con_atom) != 2:
        print("error!")
        raise
    try:
        rw_mol.AddBond(con_atom[0].GetIdx(), con_atom[1].GetIdx(), bond_type)
        rw_mol = remove_Hs(rw_mol, con_atom[0], con_atom[1], bond_label)
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        Connecting = 1 #Success
        return mol, Connecting
    except:
        Connecting = 0
        return dec_mol, Connecting