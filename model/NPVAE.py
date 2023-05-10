from rdkit.Chem import AllChem
from rdkit import Chem
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import dgl
import numpy as np
import warnings

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}
    
    def apply_node_func_enc(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

    def apply_node_func_dec(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        for ite,nd in enumerate(nodes.nodes()):
            bg_dict[int(nd)].append(h[ite])
        return {'h': h, 'c': c}

class Chem_VAE(nn.Module):
    def __init__(self, x_size, h_size, mid_size, z_dim, dropout, len_label, MAX_ITER, labels, l_1_weights, l_weights, b_weights, t_weights, m_weights, device, status, n_trial=3, test3D=False):
        super(Chem_VAE, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.z_dim = z_dim
        self.len_label=len_label
        self.MAX_ITER=MAX_ITER
        self.labels=labels
        self.device=device
        self.status=status
        self.n_trial=n_trial
        self.test3D=test3D
        
        self.l_1_weights=l_1_weights
        self.l_weights=l_weights
        self.b_weights=b_weights
        self.t_weights=t_weights
        self.m_weights=m_weights
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, mid_size)
        self.enc_mean=nn.Linear(mid_size, z_dim)
        self.enc_var=nn.Linear(mid_size, z_dim)
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        
        self.linear2=nn.Linear(x_size, h_size)
        self.linear3=nn.Linear(h_size, mid_size)
        self.linear4=nn.Linear(mid_size, mid_size)
        
        self.update_fc = nn.Linear(z_dim, z_dim)
              
        self.topo_pred=nn.Linear(z_dim, 2)
        self.label_pred1=nn.Linear(z_dim, 1024)
        self.label_pred2=nn.Linear(1024, 2048)
        self.label_pred3=nn.Linear(2048, len_label)
        
        self.bond_pred1=nn.Linear(z_dim, z_dim//2)
        self.bond_pred2=nn.Linear(z_dim//2, 3)
        
        self.label_pred1_1=nn.Linear(z_dim, 1024)
        self.label_pred1_2=nn.Linear(1024, 2048)
        self.label_pred1_3=nn.Linear(2048, len_label)
        
        self.prop_pred1=nn.Linear(z_dim, 64)
        self.prop_pred2=nn.Linear(64, 1)
        
        self.mol_pred1=nn.Linear(z_dim, mid_size)
        self.mol_pred2=nn.Linear(mid_size, h_size)
        self.mol_pred3=nn.Linear(h_size, x_size) 

    def forward(self, g, bg, bg_node_list, m_ecfp, smiles, prop, feat, h, c, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list):
        z, mean, log_var=self.encoder(g, m_ecfp, smiles, feat, h, c)        
        if self.status=="prop_train":
            topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc=\
            self.decoder(z, g, bg, bg_node_list, m_ecfp, self.z_dim, self.h_size, self.MAX_ITER,\
                         prop, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list)
            return z, mean, log_var, topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc
        elif self.status=="train":
            topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc=\
            self.decoder(z, g, bg, bg_node_list, m_ecfp, self.z_dim, self.h_size, self.MAX_ITER,\
                         prop, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list)
            return z, mean, log_var, topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc
        elif self.status=="train2D":
            topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc=\
            self.decoder(z, g, bg, bg_node_list, m_ecfp, self.z_dim, self.h_size, self.MAX_ITER,\
                         prop, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list)
            return z, mean, log_var, topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc
        elif self.status=="test":
            if self.test3D:
                dec_smi, mg, mol_pred=\
                self.decoder(z, g, bg, bg_node_list, m_ecfp, self.z_dim, self.h_size, self.MAX_ITER,\
                             prop, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list)
                return z, dec_smi, mg, mol_pred
            else:
                dec_smi, mg=\
                self.decoder(z, g, bg, bg_node_list, m_ecfp, self.z_dim, self.h_size, self.MAX_ITER,\
                             prop, root_answer, t_ans_list, b_ans_list, l_ans_list, t_id_list)
                return z, dec_smi, mg
    
    def sample_z(self, mean, log_var):
        epsilon = torch.randn(mean.shape).to(self.device)
        trick = mean + torch.exp(0.5*log_var)*epsilon
        return trick
    
    def prop_prediction(self, z, prop, loss_func_p):
        p=self.prop_pred1(z)
        p=self.prop_pred2(p)
        prop_loss = loss_func_p(p.squeeze(), prop)
        return prop_loss
    
    def encoder(self, g, m_ecfp, smiles, feat, h, c):
        g.ndata['iou'] = self.cell.W_iou(self.dropout(feat))
        g.ndata['h'] = h
        g.ndata['c'] = c
        g=dgl.reverse(g)
        dgl.prop_nodes_topo(g, reverse=False, message_func = self.cell.message_func,\
                            reduce_func = self.cell.reduce_func,\
                            apply_node_func = self.cell.apply_node_func_enc)        
        h_root = g.ndata['h'][0]
        h_root=h_root.unsqueeze(0)
        g.ndata.pop('h')
        h = self.dropout(h_root)
        y = torch.tanh(self.linear(h))
        y2=self.linear2(m_ecfp)
        y2=self.linear3(y2)
        y=self.linear4(y+y2)
        mean = self.enc_mean(y)
        log_var=self.enc_var(y)
        z = self.sample_z(mean, log_var)
        return z, mean, log_var
    
    def decoder(self, z, g, bg, bg_node_list, m_ecfp, z_dim, h_size, MAX_ITER, \
                prop, root_answer, t_ans_list, \
                b_ans_list, l_ans_list, t_id_list):
        global bg_dict
        status=self.status
        ITER=0

        if status=="test":
            if self.test3D:
                mol_pred=self.mol_prediction(z, m_ecfp, None)            
            labels=self.labels
            mg=dgl.DGLGraph()
            mg=mg.to(self.device)
            track=[]
            map_track=[]
            numnode=0
            loop=1
            while((ITER<(MAX_ITER+1)) and (loop==1)):
                if ITER==0:
                    pred, label=self.label_prediction_1(z)
                    root_smi=labels[label][0]
                    mg.add_nodes(1)
                    fp=self.make_ecfp2D(root_smi)
                    fp=np.asarray(fp)
                    fp=torch.from_numpy(fp).float()
                    feat=fp.unsqueeze(0)
                    feat=feat.to(self.device)
                    mg.ndata['ecfp']=feat
                    target_id=0
                    numnode+=1                    
                    y=self.update_z(mg, h_size, target_id, backtrack=False)
                    z=torch.tanh(self.update_fc(z+y))
                    dec_smi=copy.copy(root_smi)
                    dec_mol=self.setmap_to_mol(Chem.MolFromSmiles(dec_smi), target_id)
                    track.append(target_id)
                    map_track.append(labels[label][1])
                    
                elif ITER>0:
                    #Stop Prediction
                    p, stop_label=self.topo_prediction(z)
                    if stop_label==0: #Create a Child
                        #Bond Prediction
                        b_pred, b_label=self.bond_prediction(z)
                        #Label Prediction
                        pred, label=self.label_prediction(z)
                        new_target_id=copy.copy(numnode)

                        trial=0
                        backflag=1
                        while(trial<self.n_trial):
                            suc_smi=labels[label[trial]][0]
                            suc_mol=self.setmap_to_mol(Chem.MolFromSmiles(suc_smi), new_target_id)
                            dec_conidx=-1
                            suc_conidx=-1
                            if target_id==0:
                                for amap in map_track[-1].keys():
                                    if track.count(target_id) in map_track[-1][amap]:
                                        dec_conidx=1000*target_id+amap
                            else:
                                for amap in map_track[-1].keys():
                                    if track.count(target_id)+1 in map_track[-1][amap]:
                                        dec_conidx=1000*target_id+amap
                            for amap in labels[label[trial]][1].keys():
                                if 1 in labels[label[trial]][1][amap]:
                                    suc_conidx=1000*new_target_id+amap
                            if dec_conidx==-1:
                                break
                            if suc_conidx==-1:
                                trial+=1
                                continue
                            dec_mol, connect=self.connect_smiles(dec_mol, dec_conidx, suc_mol, suc_conidx, b_label.cpu())
                            if connect:
                                mg.add_nodes(1)
                                mg.add_edges(target_id, new_target_id)
                                fp=self.make_ecfp2D(suc_smi)
                                fp=np.asarray(fp)
                                fp=torch.from_numpy(fp).float()
                                fp=fp.unsqueeze(0)
                                feat=torch.cat((feat.cpu(),fp),dim=0)
                                feat=feat.to(self.device)
                                mg.ndata['ecfp']=feat
                                #Update
                                y=self.update_z(mg, h_size, new_target_id, backtrack=False)
                                z=torch.tanh(self.update_fc(z+y))
                                target_id=new_target_id
                                numnode+=1
                                track.append(target_id)
                                map_track.append(labels[label[trial]][1])
                                backflag=0
                                break
                            else:
                                trial+=1
                            
                        if backflag: #Backtrack
                                try:
                                    target_id=mg.predecessors(target_id).cpu()
                                    target_id=int(target_id)
                                    track.append(target_id)
                                    map_track.pop(-1)
                                    #Update
                                    y=self.update_z(mg, h_size, target_id, backtrack=True)
                                    z=torch.tanh(self.update_fc(z+y))
                                except:
                                    break                                  
                    elif stop_label==1: #STOP->Backtrack
                        if ITER==1:
                            break
                        else:
                            try:
                                target_id=mg.predecessors(target_id).cpu()
                                target_id=int(target_id)
                                track.append(target_id)
                                map_track.pop(-1)
                                #Update
                                y=self.update_z(mg, h_size, target_id, backtrack=True)
                                z=torch.tanh(self.update_fc(z+y))
                            except:
                                break
                                
                ITER+=1
            for atom in dec_mol.GetAtoms():
                dec_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0) 
            dec_smi=Chem.MolToSmiles(dec_mol)
            if self.test3D:
                return dec_smi, mg, mol_pred
            else:
                return dec_smi, mg

        elif status=="train":
            bg_dict = defaultdict(list)
            numatom=0
            topo_acc=0
            bond_acc=0
            label_acc=0
            root_acc=0
            topo_count=0
            bond_count=0
            label_count=0
            loss_func_l1=nn.CrossEntropyLoss(weight=self.l_1_weights.to(self.device))
            loss_func_l=nn.CrossEntropyLoss(weight=self.l_weights.to(self.device))
            loss_func_b=nn.CrossEntropyLoss(weight=self.b_weights.to(self.device))
            loss_func_t=nn.CrossEntropyLoss(weight=self.t_weights.to(self.device))
            loss_func_p=nn.MSELoss()
            loss_func_m=nn.BCEWithLogitsLoss(pos_weight=self.m_weights.to(self.device))
            topo_loss=0
            bond_loss=0
            label_loss=0
            nid=0

            mol_loss, _=self.mol_prediction(z, m_ecfp, loss_func_m)
            
            n = bg.number_of_nodes()
            h = torch.zeros((n, h_size)).to(self.device)
            c = torch.zeros((n, h_size)).to(self.device)
            feat=bg.ndata['ecfp']
            bg.ndata['iou'] = self.cell.W_iou(self.dropout(feat))
            bg.ndata['h'] = h
            bg.ndata['c'] = c
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            bg=dgl.reverse(bg)
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            
            while(ITER<(MAX_ITER+1)):
                if ITER==0:
                    #Root_node prediction
                    pred, label=self.label_prediction_1(z)
                    root_ans=root_answer
                    if label==root_ans:
                        root_acc=1
                    root_loss=loss_func_l1(pred, root_ans.to(self.device))
                    target_id=t_id_list.pop(0)
                    numatom+=1
                    
                    #Update
                    h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                    h_target=h_target.unsqueeze(0)
                    h = self.dropout(h_target)
                    y = torch.tanh(self.linear(h))
                    z = torch.tanh(self.update_fc(z+y))
                    nid+=1
                    
                elif ITER>0:
                    #STOP Prediction
                    p, stop_label=self.topo_prediction(z)
                    topo_count+=1
                    topo_ans=t_ans_list.pop(0)
                    if stop_label==topo_ans:
                        topo_acc+=1

                    topo_loss+=loss_func_t(p, topo_ans.to(self.device))

                    if topo_ans==1: #STOP->Backtrack
                        if ITER==1:
                            return topo_loss, None, None, root_loss, None, mol_loss, topo_acc, None, None, root_acc

                        else:
                            try:
                                target_id=t_id_list.pop(0)
                                #Update
                                h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                                h_target=h_target.unsqueeze(0)
                                h = self.dropout(h_target)
                                y = torch.tanh(self.linear(h))
                                z=torch.tanh(self.update_fc(z+y))
                                nid+=1               
                            except:
                                break
                    
                    elif topo_ans==0: #Create a child_node
                        #Bond Prediction
                        b_pred, b_label=self.bond_prediction(z)
                        bond_count+=1
                        bond_ans=b_ans_list.pop(0)
                        if b_label==bond_ans:
                            bond_acc+=1
                        bond_loss+=loss_func_b(b_pred, bond_ans.to(self.device))
                        #Label Prediction
                        pred, label=self.label_prediction(z)
                        label_count+=1
                        target_id=t_id_list.pop(0)
                        label_ans=l_ans_list.pop(0)
                        if label==label_ans:
                            label_acc+=1
                        label_loss+=loss_func_l(pred, label_ans.to(self.device))
                        numatom+=1
                        #Update
                        h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                        h_target=h_target.unsqueeze(0)
                        h = self.dropout(h_target)
                        y = torch.tanh(self.linear(h))
                        z=torch.tanh(self.update_fc(z+y))
                        nid+=1
                ITER+=1
            topo_acc=topo_acc/topo_count
            bond_acc=bond_acc/bond_count
            label_acc=label_acc/label_count
            return topo_loss, bond_loss, label_loss, root_loss, None, mol_loss, topo_acc, bond_acc, label_acc, root_acc
        
        elif status=="train2D":
            bg_dict = defaultdict(list)
            numatom=0
            topo_acc=0
            bond_acc=0
            label_acc=0
            root_acc=0
            topo_count=0
            bond_count=0
            label_count=0
            loss_func_l1=nn.CrossEntropyLoss(weight=self.l_1_weights.to(self.device))
            loss_func_l=nn.CrossEntropyLoss(weight=self.l_weights.to(self.device))
            loss_func_b=nn.CrossEntropyLoss(weight=self.b_weights.to(self.device))
            loss_func_t=nn.CrossEntropyLoss(weight=self.t_weights.to(self.device))
            loss_func_p=nn.MSELoss()
            loss_func_m=nn.BCEWithLogitsLoss(pos_weight=self.m_weights.to(self.device))
            topo_loss=0
            bond_loss=0
            label_loss=0
            nid=0
            
            n = bg.number_of_nodes()
            h = torch.zeros((n, h_size)).to(self.device)
            c = torch.zeros((n, h_size)).to(self.device)
            feat=bg.ndata['ecfp']
            bg.ndata['iou'] = self.cell.W_iou(self.dropout(feat))
            bg.ndata['h'] = h
            bg.ndata['c'] = c
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            bg=dgl.reverse(bg)
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            
            while(ITER<(MAX_ITER+1)):
                if ITER==0:
                    #Root_node prediction
                    pred, label=self.label_prediction_1(z)
                    root_ans=root_answer
                    if label==root_ans:
                        root_acc=1
                    root_loss=loss_func_l1(pred, root_ans.to(self.device))
                    target_id=t_id_list.pop(0)
                    numatom+=1
                    
                    #Update
                    h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                    h_target=h_target.unsqueeze(0)
                    h = self.dropout(h_target)
                    y = torch.tanh(self.linear(h))
                    z = torch.tanh(self.update_fc(z+y))
                    nid+=1
                    
                elif ITER>0:
                    #STOP Prediction
                    p, stop_label=self.topo_prediction(z)
                    topo_count+=1
                    topo_ans=t_ans_list.pop(0)
                    if stop_label==topo_ans:
                        topo_acc+=1

                    topo_loss+=loss_func_t(p, topo_ans.to(self.device))

                    if topo_ans==1: #STOP->Backtrack
                        if ITER==1:
                            return topo_loss, None, None, root_loss, None, None, topo_acc, None, None, root_acc

                        else:
                            try:
                                target_id=t_id_list.pop(0)
                                #Update
                                h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                                h_target=h_target.unsqueeze(0)
                                h = self.dropout(h_target)
                                y = torch.tanh(self.linear(h))
                                z=torch.tanh(self.update_fc(z+y))
                                nid+=1              
                            except:
                                break
                    
                    elif topo_ans==0: #Create a child_node
                        #Bond Prediction
                        b_pred, b_label=self.bond_prediction(z)
                        bond_count+=1
                        bond_ans=b_ans_list.pop(0)
                        if b_label==bond_ans:
                            bond_acc+=1
                        bond_loss+=loss_func_b(b_pred, bond_ans.to(self.device))
                        #Label Prediction
                        pred, label=self.label_prediction(z)
                        label_count+=1
                        target_id=t_id_list.pop(0)
                        label_ans=l_ans_list.pop(0)
                        if label==label_ans:
                            label_acc+=1
                        label_loss+=loss_func_l(pred, label_ans.to(self.device))
                        numatom+=1
                        #Update
                        h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                        h_target=h_target.unsqueeze(0)
                        h = self.dropout(h_target)
                        y = torch.tanh(self.linear(h))
                        z=torch.tanh(self.update_fc(z+y))
                        nid+=1
                ITER+=1
            topo_acc=topo_acc/topo_count
            bond_acc=bond_acc/bond_count
            label_acc=label_acc/label_count
            return topo_loss, bond_loss, label_loss, root_loss, None, None, topo_acc, bond_acc, label_acc, root_acc

        elif status=="prop_train":
            bg_dict = defaultdict(list)
            numatom=0
            topo_acc=0
            bond_acc=0
            label_acc=0
            root_acc=0
            topo_count=0
            bond_count=0
            label_count=0
            loss_func_l1=nn.CrossEntropyLoss(weight=self.l_1_weights.to(self.device))
            loss_func_l=nn.CrossEntropyLoss(weight=self.l_weights.to(self.device))
            loss_func_b=nn.CrossEntropyLoss(weight=self.b_weights.to(self.device))
            loss_func_t=nn.CrossEntropyLoss(weight=self.t_weights.to(self.device))
            loss_func_p=nn.MSELoss()
            loss_func_m=nn.BCEWithLogitsLoss(pos_weight=self.m_weights.to(self.device))
            topo_loss=0
            bond_loss=0
            label_loss=0
            nid=0
            if prop is not None:
                prop_loss=self.prop_prediction(z, prop, loss_func_p)
            mol_loss, _=self.mol_prediction(z, m_ecfp, loss_func_m)
            
            n = bg.number_of_nodes()
            h = torch.zeros((n, h_size)).to(self.device)
            c = torch.zeros((n, h_size)).to(self.device)
            feat=bg.ndata['ecfp']
            bg.ndata['iou'] = self.cell.W_iou(self.dropout(feat))
            bg.ndata['h'] = h
            bg.ndata['c'] = c
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            bg=dgl.reverse(bg)
            dgl.prop_nodes_topo(bg, reverse=False, message_func = self.cell.message_func,\
                                reduce_func = self.cell.reduce_func,\
                                apply_node_func = self.cell.apply_node_func_dec)
            
            while(ITER<(MAX_ITER+1)):
                if ITER==0:
                    #Root_node prediction
                    pred, label=self.label_prediction_1(z)
                    root_ans=root_answer
                    if label==root_ans:
                        root_acc=1
                    root_loss=loss_func_l1(pred, root_ans.to(self.device))
                    target_id=t_id_list.pop(0)
                    numatom+=1
                    
                    #Update
                    h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                    h_target=h_target.unsqueeze(0)
                    h = self.dropout(h_target)
                    y = torch.tanh(self.linear(h))
                    z = torch.tanh(self.update_fc(z+y))
                    nid+=1
                    
                elif ITER>0:
                    #STOP Prediction
                    p, stop_label=self.topo_prediction(z)
                    topo_count+=1
                    topo_ans=t_ans_list.pop(0)
                    if stop_label==topo_ans:
                        topo_acc+=1

                    topo_loss+=loss_func_t(p, topo_ans.to(self.device))

                    if topo_ans==1: #STOP->Backtrack
                        if ITER==1:
                            if prop is None:
                                return topo_loss, None, None, root_loss, None, mol_loss, topo_acc, None, None, root_acc
                            else:
                                return topo_loss, None, None, root_loss, prop_loss, mol_loss, topo_acc, None, None, root_acc

                        else:
                            try:
                                target_id=t_id_list.pop(0)
                                #Update
                                h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                                h_target=h_target.unsqueeze(0)
                                h = self.dropout(h_target)
                                y = torch.tanh(self.linear(h))
                                z=torch.tanh(self.update_fc(z+y))
                                nid+=1
                            except:
                                break
                    
                    elif topo_ans==0: #Create a child_node
                        #Bond Prediction
                        b_pred, b_label=self.bond_prediction(z)
                        bond_count+=1
                        bond_ans=b_ans_list.pop(0)
                        if b_label==bond_ans:
                            bond_acc+=1
                        bond_loss+=loss_func_b(b_pred, bond_ans.to(self.device))
                        #Label Prediction
                        pred, label=self.label_prediction(z)
                        label_count+=1
                        target_id=t_id_list.pop(0)
                        label_ans=l_ans_list.pop(0)
                        if label==label_ans:
                            label_acc+=1
                        label_loss+=loss_func_l(pred, label_ans.to(self.device))
                        numatom+=1
                        #Update
                        h_target = bg_dict[bg_node_list[nid][0]][bg_node_list[nid][1]]
                        h_target=h_target.unsqueeze(0)
                        h = self.dropout(h_target)
                        y = torch.tanh(self.linear(h))
                        z=torch.tanh(self.update_fc(z+y))
                        nid+=1
                ITER+=1
            topo_acc=topo_acc/topo_count
            bond_acc=bond_acc/bond_count
            label_acc=label_acc/label_count
            if prop is None:
                return topo_loss, bond_loss, label_loss, root_loss, None, mol_loss, topo_acc, bond_acc, label_acc, root_acc
            else:
                return topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, topo_acc, bond_acc, label_acc, root_acc
    
    def make_ecfp2D(self, smiles, n_bit=2048, r=2):
        mol=Chem.MolFromSmiles(smiles)
        ecfp=AllChem.GetMorganFingerprintAsBitVect(mol, r, n_bit, useChirality=False)
        return ecfp

    def topo_prediction(self, h):
        p=self.topo_pred(h)
        if torch.argmax(p)==1:
            stop_label=1 #STOP
        elif torch.argmax(p)==0:
            stop_label=0 #CONTINUE
        return p, stop_label
    
    def bond_prediction(self, h):
        b_pred=torch.tanh(self.bond_pred1(h))
        b_pred=self.bond_pred2(b_pred)
        b_label=torch.argmax(b_pred)
        return b_pred, b_label
    
    def label_prediction(self, h):
        y=torch.tanh(self.label_pred1(h))
        y=torch.tanh(self.label_pred2(y))
        pred=self.label_pred3(y)
        if self.status!='test':
            label=torch.argmax(pred)
            return pred, label
        else:
            label=torch.argsort(pred[0], descending=True)[:self.n_trial]
            return pred, label
    
    def label_prediction_1(self, h):
        y=torch.tanh(self.label_pred1_1(h))
        y=torch.tanh(self.label_pred1_2(y))
        pred=self.label_pred1_3(y)
        label=torch.argmax(pred)
        return pred, label
    
    def mol_prediction(self, z, m_ecfp, loss_func_m):
        y=torch.tanh(self.mol_pred1(z))
        y=torch.tanh(self.mol_pred2(y))
        y=self.mol_pred3(y)
        if self.status!='test':
            mol_loss=loss_func_m(y, m_ecfp)
            return mol_loss, F.sigmoid(y)
        else:
            return F.sigmoid(y)
        
    def update_z(self, tree, h_size, target_id, backtrack):
        tree=tree.to(torch.device(self.device))
        n = tree.number_of_nodes()
        h = torch.zeros((n, h_size)).to(self.device)
        c = torch.zeros((n, h_size)).to(self.device)
        feat=tree.ndata['ecfp']
        tree.ndata['iou'] = self.cell.W_iou(self.dropout(feat))
        tree.ndata['h'] = h
        tree.ndata['c'] = c
        if backtrack==True:
            dgl.prop_nodes_topo(tree, reverse=False, message_func = self.cell.message_func,
                            reduce_func = self.cell.reduce_func,
                            apply_node_func = self.cell.apply_node_func_enc)
            tree=dgl.reverse(tree)
            dgl.prop_nodes_topo(tree, reverse=False, message_func = self.cell.message_func,
                            reduce_func = self.cell.reduce_func,
                            apply_node_func = self.cell.apply_node_func_enc)
            h_target = tree.ndata['h'][target_id]
            h_target=h_target.unsqueeze(0)
            tree.ndata.pop('h')
            h = self.dropout(h_target)
            y = torch.tanh(self.linear(h))
            return y

        elif backtrack==False:
            dgl.prop_nodes_topo(tree, reverse=False, message_func = self.cell.message_func,
                            reduce_func = self.cell.reduce_func,
                            apply_node_func = self.cell.apply_node_func_enc)    
            h_target = tree.ndata['h'][target_id]
            h_target=h_target.unsqueeze(0)
            tree.ndata.pop('h')
            h = self.dropout(h_target)
            y = torch.tanh(self.linear(h))
            return y

    def get_smiles(self, mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
        
    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: Chem.Kekulize(mol)
        return mol

    def sanitize(self, mol, kekulize=True):
        try:
            smiles = self.get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
            mol = self.get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
        except:
            mol = None
        return mol
           
    def setmap_to_mol(self, mol, node_id):
        for atom in mol.GetAtoms():
            mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(node_id*1000+atom.GetIdx())
        return mol

    def connect_smiles(self, dec_mol, dec_conidx, suc_mol, suc_conidx, bond_label):
        warnings.simplefilter('ignore')
        if bond_label==0:
            bond_type=Chem.BondType.SINGLE
        elif bond_label==1:
            bond_type=Chem.BondType.DOUBLE
        elif bond_label==2:
            bond_type=Chem.BondType.TRIPLE
        else:
            raise
        con_smi=Chem.MolToSmiles(dec_mol)+'.'+Chem.MolToSmiles(suc_mol)
        con_mol=Chem.MolFromSmiles(con_smi)
        rw_mol=Chem.RWMol(con_mol)
        con_atom=[]
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomMapNum()==dec_conidx or atom.GetAtomMapNum()==suc_conidx:
                con_atom.append(atom)
        if len(con_atom)!=2:
            Connecting=0
            return dec_mol, Connecting
        try:
            rw_mol.AddBond(con_atom[0].GetIdx(), con_atom[1].GetIdx(), bond_type)
            rw_mol=self.remove_Hs(rw_mol, con_atom[0], con_atom[1], bond_label)
            mol = rw_mol.GetMol()
            Chem.SanitizeMol(mol)
            Connecting=1
        except:
            if bond_label==0:
                Connecting=0
                return dec_mol, Connecting
            else:
                mol, Connecting=self.connect_smiles(dec_mol, dec_conidx, suc_mol, suc_conidx, bond_label-1)
        return mol, Connecting

    def remove_Hs(self, rwmol, a1, a2, bond_label):
        if bond_label==0:
            num=1
        elif bond_label==1:
            num=2
        elif bond_label==2:
            num=3
        else:
            raise
        rwmol=Chem.AddHs(rwmol)
        rwmol=Chem.RWMol(rwmol)
        h_map1=2000000
        h_map2=3000000
        f_h_map1=copy.copy(h_map1)
        f_h_map2=copy.copy(h_map2)
        for b in rwmol.GetBonds():
            s_atom = b.GetBeginAtom()
            e_atom = b.GetEndAtom()
            if (e_atom.GetIdx()==a1.GetIdx()) and (s_atom.GetSymbol()=='H'):
                s_atom.SetAtomMapNum(h_map1)
                h_map1+=1
            elif (s_atom.GetIdx()==a1.GetIdx()) and (e_atom.GetSymbol()=='H'):
                e_atom.SetAtomMapNum(h_map1)
                h_map1+=1
            elif (e_atom.GetIdx()==a2.GetIdx()) and (s_atom.GetSymbol()=='H'):
                s_atom.SetAtomMapNum(h_map2)
                h_map2+=1
            elif (s_atom.GetIdx()==a2.GetIdx()) and (e_atom.GetSymbol()=='H'):
                e_atom.SetAtomMapNum(h_map2)
                h_map2+=1
        for i in range(num):
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum()==f_h_map1+i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum()==f_h_map2+i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
        rwmol=rwmol.GetMol()
        rwmol=self.sanitize(rwmol, kekulize=False)
        rwmol=Chem.RemoveHs(rwmol)
        rwmol=Chem.RWMol(rwmol)
        return rwmol