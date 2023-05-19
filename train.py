from __future__ import print_function
import os
import mkl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from rdkit import Chem
from rdkit.Chem import Descriptors
import dgl
from collections import OrderedDict
import pickle
import copy
import datetime
import model.NPVAE as NPVAE
import numpy as np
import npscorer
import gzip
import warnings

def main(rank, num_gpu, batchsize, smiles_path, prepared_path, x_size, h_size, mid_size, z_dim, dropout, max_iter, max_epoch, load_epoch, load_path, warmup_epoch, anneal_epoch, temp, kl_m, t_m, b_m, l_1_m, l_m, p_m, m_m, save_epoch, save_path, device, prop_info, status):
    warnings.simplefilter('ignore')
    dist.init_process_group(                                   
    backend = 'nccl',                                         
    init_method = 'env://',   
    world_size = num_gpu,
    rank = rank                                               
    )                                                          
    torch.manual_seed(0)
    #Loading
    with open(smiles_path) as f:
        all_smiles = [line.strip("\r\n ").split()[0] for line in f]
    with open(prepared_path + "/input_data/graphs", "rb")as f:
        graphs = pickle.load(f)
    with open(prepared_path + "/input_data/sub_trees", "rb")as f:
        sub_trees = pickle.load(f)
    with open(prepared_path + "/input_data/labels", "rb") as f:
        labels = pickle.load(f)
    with open(prepared_path + "/input_data/weights/t_weights", "rb")as f:
        t_weights = pickle.load(f)
    with open(prepared_path + "/input_data/weights/b_weights", "rb")as f:
        b_weights = pickle.load(f)
    with open(prepared_path + "/input_data/weights/l_weights", "rb")as f:
        l_weights  =pickle.load(f)
    with open(prepared_path + "/input_data/weights/l_1_weights", "rb")as f:
        l_1_weights = pickle.load(f)
    with open(prepared_path + "/input_data/ecfp_list3D", "rb")as f:
        ecfp_list3D = pickle.load(f)
    with open(prepared_path + "/input_data/weights/m_weights", "rb")as f:
        m_weights = pickle.load(f)
    with open(prepared_path + "/input_data/bg_node_list", "rb")as f:
        bg_node_list = pickle.load(f)
    with open(prepared_path + "/input_data/root_ans_list", "rb")as f:
        root_ans_list = pickle.load(f)
    with open(prepared_path + "/input_data/label_ans_list", "rb")as f:
        label_ans_list = pickle.load(f)
    with open(prepared_path + "/input_data/bond_ans_list", "rb")as f:
        bond_ans_list = pickle.load(f)
    with open(prepared_path + "/input_data/topo_ans_list", "rb")as f:
        topo_ans_list = pickle.load(f)
    with open(prepared_path + "/input_data/target_id_list", "rb")as f:
        target_id_list = pickle.load(f)

    #Making batched graphs
    bg_list = []
    for i in range(len(sub_trees)):
        bg_list.append(dgl.batch(sub_trees[i]))
        
    if prop_info == "logp":
        logp_list = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in all_smiles]
        dataset = []
        for i in range(len(graphs)):
            dataset.append((graphs[i], bg_list[i], bg_node_list[i], ecfp_list3D[i], \
                            all_smiles[i], graphs[i].ndata['ecfp'], logp_list[i], \
                            root_ans_list[i], topo_ans_list[i], bond_ans_list[i], \
                            label_ans_list[i], target_id_list[i]))
        del graphs
        del bg_list
        del bg_node_list
        del ecfp_list3D
        del logp_list
        del sub_trees
        del root_ans_list
        del topo_ans_list
        del bond_ans_list
        del label_ans_list
        del target_id_list
    elif prop_info == "nplikeness":
        fs = pickle.load(gzip.open('./publicnp.model.gz'))
        nplikeness_list = [npscorer.scoreMol(Chem.MolFromSmiles(s), fs) for s in all_smiles]
        dataset = []
        for i in range(len(graphs)):
            dataset.append((graphs[i], bg_list[i], bg_node_list[i], ecfp_list3D[i], \
                            all_smiles[i], graphs[i].ndata['ecfp'], nplikeness_list[i], \
                            root_ans_list[i], topo_ans_list[i], bond_ans_list[i], \
                            label_ans_list[i], target_id_list[i]))
        del graphs
        del bg_list
        del bg_node_list
        del ecfp_list3D
        del nplikeness_list
        del sub_trees
        del root_ans_list
        del topo_ans_list
        del bond_ans_list
        del label_ans_list
        del target_id_list
    elif prop_info is None:
        dataset = []
        for i in range(len(graphs)):
            dataset.append((graphs[i], bg_list[i], bg_node_list[i], ecfp_list3D[i], \
                            all_smiles[i], graphs[i].ndata['ecfp'], None, \
                            root_ans_list[i], topo_ans_list[i], bond_ans_list[i], \
                            label_ans_list[i], target_id_list[i]))
        del graphs
        del bg_list
        del bg_node_list
        del ecfp_list3D
        del sub_trees
        del root_ans_list
        del topo_ans_list
        del bond_ans_list
        del label_ans_list
        del target_id_list        
    else:
        try:
            with open(prepared_path + "/input_data/prop_info", "rb")as f:
                prop_info = pickle.load(f)
        except:
            raise Exception("Please prepare a pickle file 'prop_info' under the input_data directory.")
        if len(prop_info) != len(graphs):
            raise Exception("The number of input compounds and the number of property values do not match. Enter 'None' for compounds with unknown function values.")
        else:
            dataset = []
            for i in range(len(graphs)):
                dataset.append((graphs[i], bg_list[i], bg_node_list[i], ecfp_list3D[i], \
                                all_smiles[i], graphs[i].ndata['ecfp'], prop_info[i], \
                                root_ans_list[i], topo_ans_list[i], bond_ans_list[i], \
                                label_ans_list[i], target_id_list[i]))
            del graphs
            del bg_list
            del bg_node_list
            del ecfp_list3D
            del logp_list
            del nplikeness_list
            del sub_trees
            del root_ans_list
            del topo_ans_list
            del bond_ans_list
            del label_ans_list
            del target_id_list
            
    sampler = DistributedSampler(dataset, num_replicas = num_gpu, rank = rank, shuffle = True)
    data_loader = DataLoader(dataset, batch_size = batchsize, shuffle = False, pin_memory = True, collate_fn = collate, num_workers = 0, sampler = sampler)
    del dataset
    torch.cuda.set_device(rank)
    
    model = NPVAE.Chem_VAE(x_size, h_size, mid_size, z_dim, dropout, len(labels), max_iter, labels, l_1_weights, l_weights, b_weights, t_weights, m_weights, device, status)
    
   #Loading parameters
    if load_epoch > 0:
        load_path = load_path + '/model.iter-{}'.format(load_epoch)
    else:
        load_path = 0
    if load_path != 0:
        dist.barrier()
        map_location = {'cuda:0': 'cuda:%d'%rank}
        state_dict = torch.load(load_path, map_location = map_location)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.to(rank)
    
    # Wrap the model
    model = DDP(model, device_ids = [rank], find_unused_parameters = True)
    model.train()

    # define loss function (criterion) and optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    
    dt_now = datetime.date.today()
    filename = save_path + '/Log_{}_{}.txt'.format(os.path.basename(__name__), dt_now)
    if rank == 0:
        with open(filename, 'a') as f:
            f.write("{} / {} / rank:{}\n".format(os.path.basename(__name__), datetime.datetime.now(), rank))
        start = time.time()
    for epoch in range(max_epoch):
        sampler.set_epoch(epoch)
        if rank == 0:
            with open(filename, 'a') as f:
                f.write("Epoch:{}\n".format(epoch + 1))
        kl_losses = []
        topo_losses = []
        bond_losses = []
        label_losses = []
        root_losses = []
        prop_losses = []
        mol_losses = []
        all_losses = []
        topo_a = []
        bond_a = []
        label_a = []
        root_a = []
        for iter, (g, bg, bg_node, m_ecfp, smiles, feats, prop, root_answer, topo_ans_list, \
                   bond_ans_list, label_ans_list, target_id_list) in enumerate(data_loader):
            g = g.to('cuda:{}'.format(rank))
            bg = bg[0].to('cuda:{}'.format(rank))
            bg_node = bg_node[0]
            feats = feats[0].to(device)
            m_ecfp = m_ecfp[0].to(device)
            smiles = smiles[0]
            feat = g.ndata['ecfp']
            if prop is not None:
                prop = torch.tensor([prop[0]])
                prop = prop.unsqueeze(0)
                prop = prop.to(device)
            root_answer = root_answer[0]
            t_ans_list = copy.copy(topo_ans_list[0])
            b_ans_list = copy.copy(bond_ans_list[0])
            l_ans_list = copy.copy(label_ans_list[0])
            t_id_list = copy.copy(target_id_list[0])
            n = g.number_of_nodes()
            h = torch.zeros((n, h_size)).to(device)
            c = torch.zeros((n, h_size)).to(device)
            z, mean, log_var, topo_loss, bond_loss, label_loss, root_loss, prop_loss, mol_loss, \
            topo_acc, bond_acc, label_acc, root_acc = model(g, bg, bg_node, m_ecfp, \
                                                            smiles, prop, feats, h, c, \
                                                            root_answer, t_ans_list, b_ans_list, \
                                                            l_ans_list, t_id_list)
            #Calculate Loss
            kl_loss = -torch.sum(0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)))
            kl_losses.append(kl_loss.detach().item())
            if status == "prop_train":
                if bond_loss == None: #When the number of nodes is 1
                    if prop is None:
                        loss = t_m * topo_loss + l_1_m * root_loss + kl_m * kl_loss + m_m * mol_loss
                        topo_losses.append(topo_loss.detach().item())
                        root_losses.append(root_loss.detach().item())
                        mol_losses.append(mol_loss.detach().item())
                        all_losses.append(loss.detach().item())
                        topo_a.append(topo_acc)
                        root_a.append(root_acc)
                    else:
                        loss = t_m * topo_loss + l_1_m * root_loss + kl_m * kl_loss + p_m * prop_loss + m_m * mol_loss
                        topo_losses.append(topo_loss.detach().item())
                        root_losses.append(root_loss.detach().item())
                        prop_losses.append(prop_loss.detach().item())
                        mol_losses.append(mol_loss.detach().item())
                        all_losses.append(loss.detach().item())
                        topo_a.append(topo_acc)
                        root_a.append(root_acc)
                else:
                    if prop is None:
                        loss = t_m * topo_loss + b_m * bond_loss + l_m * label_loss + l_1_m * root_loss + kl_m * kl_loss + m_m * mol_loss
                        topo_losses.append(topo_loss.detach().item())
                        bond_losses.append(bond_loss.detach().item())
                        label_losses.append(label_loss.detach().item())
                        root_losses.append(root_loss.detach().item())
                        mol_losses.append(mol_loss.detach().item())
                        all_losses.append(loss.detach().item())
                        topo_a.append(topo_acc)
                        bond_a.append(bond_acc)
                        label_a.append(label_acc)
                        root_a.append(root_acc)
                    else:
                        loss = t_m * topo_loss + b_m * bond_loss + l_m * label_loss + l_1_m * root_loss + kl_m * kl_loss + p_m * prop_loss + m_m * mol_loss
                        topo_losses.append(topo_loss.detach().item())
                        bond_losses.append(bond_loss.detach().item())
                        label_losses.append(label_loss.detach().item())
                        root_losses.append(root_loss.detach().item())
                        prop_losses.append(prop_loss.detach().item())
                        mol_losses.append(mol_loss.detach().item())
                        all_losses.append(loss.detach().item())
                        topo_a.append(topo_acc)
                        bond_a.append(bond_acc)
                        label_a.append(label_acc)
                        root_a.append(root_acc)
            elif status == "train2D":
                if bond_loss == None: #When the number of nodes is 1
                    loss = t_m * topo_loss + l_1_m * root_loss + kl_m * kl_loss
                    topo_losses.append(topo_loss.detach().item())
                    root_losses.append(root_loss.detach().item())
                    all_losses.append(loss.detach().item())
                    topo_a.append(topo_acc)
                    root_a.append(root_acc)
                else:
                    loss = t_m * topo_loss + b_m * bond_loss + l_m * label_loss + l_1_m * root_loss + kl_m * kl_loss
                    topo_losses.append(topo_loss.detach().item())
                    bond_losses.append(bond_loss.detach().item())
                    label_losses.append(label_loss.detach().item())
                    root_losses.append(root_loss.detach().item())
                    all_losses.append(loss.detach().item())
                    topo_a.append(topo_acc)
                    bond_a.append(bond_acc)
                    label_a.append(label_acc)
                    root_a.append(root_acc)
            elif status == "train":
                if bond_loss == None: #When the number of nodes is 1
                    loss = t_m * topo_loss + l_1_m * root_loss + kl_m * kl_loss + m_m * mol_loss
                    topo_losses.append(topo_loss.detach().item())
                    root_losses.append(root_loss.detach().item())
                    mol_losses.append(mol_loss.detach().item())
                    all_losses.append(loss.detach().item())
                    topo_a.append(topo_acc)
                    root_a.append(root_acc)
                else:
                    loss = t_m * topo_loss + b_m * bond_loss + l_m * label_loss + l_1_m * root_loss + kl_m * kl_loss + m_m * mol_loss
                    topo_losses.append(topo_loss.detach().item())
                    bond_losses.append(bond_loss.detach().item())
                    label_losses.append(label_loss.detach().item())
                    root_losses.append(root_loss.detach().item())
                    mol_losses.append(mol_loss.detach().item())
                    all_losses.append(loss.detach().item())
                    topo_a.append(topo_acc)
                    bond_a.append(bond_acc)
                    label_a.append(label_acc)
                    root_a.append(root_acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        kl_l = sum(kl_losses) / len(kl_losses)
        t_l = sum(topo_losses) / len(topo_losses)
        b_l = sum(bond_losses) / len(bond_losses)
        l_l = sum(label_losses) / len(label_losses)
        r_l = sum(root_losses) / len(root_losses)
        if status == "prop_train" and prop is not None:
            p_l = sum(prop_losses) / len(prop_losses)
        if status == 'prop_train' or status == "train":
            m_l = sum(mol_losses) / len(mol_losses)
        t_a = sum(topo_a) / len(topo_a)
        b_a = sum(bond_a) / len(bond_a)
        l_a = sum(label_a) / len(label_a)
        r_a = sum(root_a) / len(root_a)
        a_l = sum(all_losses) / len(all_losses)

        #Save
        if (epoch + 1) % save_epoch == 0:
            dist.barrier()
            if rank == 0:
                torch.save(model.state_dict(), save_path + "/model.iter-" + str(load_epoch + epoch + 1))
        
        if rank == 0:
            if status == "prop_train":
                if prop is None:
                    with open(filename, 'a') as f:
                        f.write("[{}] <Loss> KL:{}, Root:{}, Topo:{}, Bond:{}, Label:{}, Mol:{} <ACC> Root:{}, Topo:{}, Bond:{}, Label:{}\n".format((epoch + 1), kl_l, r_l, t_l, b_l, l_l, m_l, r_a, t_a, b_a, l_a))   
                else:
                    with open(filename, 'a') as f:
                        f.write("[{}] <Loss> KL:{}, Root:{}, Topo:{}, Bond:{}, Label:{}, Prop:{}, Mol:{} <ACC> Root:{}, Topo:{}, Bond:{}, Label:{}\n".format((epoch + 1), kl_l, r_l, t_l, b_l, l_l, p_l, m_l, r_a, t_a, b_a, l_a))   
            elif status == "train":
                with open(filename, 'a') as f:
                    f.write("[{}] <Loss> KL:{}, Root:{}, Topo:{}, Bond:{}, Label:{}, Mol:{} <ACC> Root:{}, Topo:{}, Bond:{}, Label:{}\n".format((epoch + 1), kl_l, r_l, t_l, b_l, l_l, m_l, r_a, t_a, b_a, l_a))
            elif status == "train2D":
                with open(filename, 'a') as f:
                    f.write("[{}] <Loss> KL:{}, Root:{}, Topo:{}, Bond:{}, Label:{} <ACC> Root:{}, Topo:{}, Bond:{}, Label:{}\n".format((epoch + 1), kl_l, r_l, t_l, b_l, l_l, r_a, t_a, b_a, l_a))         
        if (warmup_epoch != -1) and (anneal_epoch != -1):
            if (epoch + 1) > warmup_epoch and (epoch + 1) % anneal_epoch == 0:
                kl_m += temp
                if rank == 0:
                    with open(filename, 'a') as f:
                        f.write("KL coef:{}\n".format(kl_m))               
    if rank == 0:
        end = time.time()
        with open(filename, 'a') as f:
            f.write("Learning time: {}\n".format(end - start))
            
def collate(samples):
    graphs, bg, bg_node, m_ecfp, smiles, feats, prop, root_answer, \
    topo_ans_list, bond_ans_list, label_ans_list, target_id_list = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, bg, bg_node, m_ecfp, smiles, feats, prop, root_answer, topo_ans_list, bond_ans_list, label_ans_list, target_id_list
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ngpu", "--num_gpu", type = int, choices = range(0, torch.cuda.device_count() + 1),
                        default = torch.cuda.device_count(), help = "Number of distributed GPUs")
    parser.add_argument("-batch", "--batchsize", type = int,
                        default = 1, help = "Batch size")
    parser.add_argument("--smiles_path", type = str,
                        default = "./smiles_data/drugbank_smiles.txt", help = "Path of SMILES data for input compounds")
    parser.add_argument("--prepared_path", type = str,
                        default = "./save_data", help = "The path where the created 'input data' is saved")
    parser.add_argument("--x_size", type = int,
                        default = 2048, help = "Dimension of the input (ECFP)")
    parser.add_argument("--h_size", type = int,
                        default = 512, help = "Dimension of the hidden layer")
    parser.add_argument("--mid_size", type = int,
                        default = 256, help = "Dimension of the middle layer")
    parser.add_argument("--z_dim", type = int,
                        default = 256, help = "Dimension of the latent variable")
    parser.add_argument("--dropout", type = int,
                        default = 0, help  = "Dropout rate of Tree-LSTM")
    parser.add_argument("--max_iter", type = int,
                        default = 500, help = "Maximum number of iterations in the decoding process (should be set to a large value when dealing with huge molecules)")
    parser.add_argument("--max_epoch", type = int,
                        default = 100, help = "Number of epochs")
    parser.add_argument("--load_epoch", type = int,
                        default = 0, help = "Epoch specified for loading parameters")
    parser.add_argument("--load_path", type = str,
                        default = "./param_data", help = "The path where learned parameters are saved (Please set if you want to continue learning additional)")
    parser.add_argument("--warmup_epoch", type = int,
                        default = -1, help = "Number of initial epochs that do not change the coefficient of KL loss (-1 means no setting)")
    parser.add_argument("--anneal_epoch", type = int,
                        default = -1, help = "Epoch intervals to change the coefficient of KL loss (-1 means no setting)")
    parser.add_argument("-temp", "--temperature", type = float,
                        default = 0.001, help = "Amount of change in KL loss (Meaningless value when WARMUP_EPOCH or ANNEAL_EPOCH is -1)")
    parser.add_argument("-kl_m", "--kl_magnification", type = float,
                        default = 0.01, help = "Coefficient of KL loss")
    parser.add_argument("-t_m", "--topology_magnification", type = float,
                        default = 3.0, help = "Coefficient of topology loss")
    parser.add_argument("-b_m", "--bond_magnification", type = float,
                        default = 1.0, help = "Coefficient of bond loss")
    parser.add_argument("-l_1_m", "--root_magnification", type = float,
                        default = 2.0, help = "Coefficient of root label loss")
    parser.add_argument("-l_m", "--label_magnification", type = float,
                        default = 1.0, help = "Coefficient of label loss")
    parser.add_argument("-p_m", "--prop_magnification", type = float,
                        default = 1.0, help = "Coefficient of property loss")
    parser.add_argument("-m_m", "--mol_magnification", type = float,
                        default = 2.0, help = "Coefficient of mol (conformation) loss")
    parser.add_argument("--save_epoch", type = int,
                        default = 20, help = "Epoch intervals to save parameters")
    parser.add_argument("--save_path", type = str,
                        default = "./param_data", help = "Path to save parameters")
    parser.add_argument("--device", type = str,
                        default = 'cuda' if torch.cuda.is_available() else 'cpu', help = "Whether the device is GPU or CPU")
    parser.add_argument("--prop_info", type = str,
                        default = 'nplikeness', help = "Functional information used for learning")
    parser.add_argument("--status", choices = ['prop_train', 'train', 'train2D'],
                        default = "prop_train", help = "Learning mode (Select from 'prop_train', 'train', 'train2D')")
    args = parser.parse_args()
    print("Whether GPU is available:", torch.cuda.is_available())
    print("Whether DDP is available:", torch.distributed.is_available())
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    parameters = (args.num_gpu, args.batchsize, args.smiles_path, args.prepared_path, args.x_size, args.h_size, args.mid_size, args.z_dim, args.dropout, args.max_iter, args.max_epoch, args.load_epoch, args.load_path, args.warmup_epoch, args.anneal_epoch, args.temperature, args.kl_magnification, args.topology_magnification, args.bond_magnification, args.root_magnification, args.label_magnification, args.prop_magnification, args.mol_magnification, args.save_epoch, args.save_path, args.device, args.prop_info, args.status)
    print("Now Training...")
    mp.spawn(main, nprocs = args.num_gpu, args = parameters, join = True)
    print("Finished")