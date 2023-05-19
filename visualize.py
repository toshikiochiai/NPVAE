from __future__ import print_function
import pickle
import gzip
import npscorer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import RDLogger, Chem
import warnings

def main(args):
    warnings.simplefilter('ignore')
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    #Loading
    with open(args.smiles_path) as f:
        all_smiles = [line.strip("\r\n ").split()[0] for line in f]
    with open(args.saved_path + "/z_list", "rb")as f:
        z_list = pickle.load(f)
    try:
        with open(args.check_smiles_path) as f:
            check_smiles = [line.strip("\r\n ").split()[0] for line in f]
    except:
        check_smiles = []
        print("No 'check_smiles' exists, so it does not plot the location of specific compounds")
    if args.color_code:
        if args.prop_info == "logp":
            prop_list = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in all_smiles]    
        elif args.prop_info == "nplikeness":
            fs = pickle.load(gzip.open('./publicnp.model.gz'))
            prop_list = [npscorer.scoreMol(Chem.MolFromSmiles(s), fs) for s in all_smiles]
        else:
            try:
                with open(args.prepared_path + "/input_data/prop_info", "rb")as f:
                    prop_list = pickle.load(f)
            except:
                raise Exception("Please prepare a pickle file 'prop_info' under the input_data directory.")
            if len(prop_list) != len(all_smiles):
                raise Exception("The number of input compounds and the number of property values do not match. Enter 'None' for compounds with unknown function values.")
        q1, q2 = np.percentile(prop_list, [3 ,97])
    #tSNE
    z_list_np = np.array(z_list)
    z_list_np = z_list_np.reshape(z_list_np.shape[0], z_list_np.shape[2])
    tsne = TSNE(n_components = 2, perplexity = args.perplexity, n_iter = args.n_iter, random_state = args.random_state)
    z_tsne = tsne.fit_transform(z_list_np)
    if len(check_smiles) != 0:
        index = []
        for s in check_smiles:
            try:
                idx = all_smiles.index(s)
                index.append(idx)
            except:
                raise Exception("'check_smiles' must be included in the SMILES data in the original 'smiles_path'")
    #Plot
    if args.color_code and len(check_smiles) == 0:
        label = prop_list
        plt.figure(figsize = (12, 9))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s = 20, alpha = 0.8, marker = 'o', edgecolors = 'none', cmap = 'viridis', c = prop_list, vmin = q1, vmax = q2)
        plt.colorbar()
        plt.title("Latent Space")
        plt.savefig(args.saved_path + "/visualization.png")
    elif args.color_code and len(check_smiles) != 0:
        label = prop_list
        plt.figure(figsize = (12, 9))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s = 20, alpha = 0.8, marker = 'o', edgecolors = 'none', cmap = 'viridis', c = prop_list, vmin = q1, vmax = q2)
        plt.colorbar()
        plt.title("Latent Space")
        for n in index:
            plt.scatter(z_tsne[n, 0], z_tsne[n, 1], s = 80, alpha = 0.9, marker = 'o', edgecolors = 'black', color = 'red')  
        plt.savefig(args.saved_path + "/visualization.png")
    elif args.color_code == False and len(check_smiles) != 0:
        plt.figure(figsize = (12, 10))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s = 20, alpha = 0.8, marker = 'o', edgecolors = 'none', color = 'darkgray')
        plt.title("Latent Space")
        for n in index:
            plt.scatter(z_tsne[n, 0], z_tsne[n, 1], s = 80, alpha = 0.9, marker = 'o', edgecolors = 'black', color = 'red')  
        plt.savefig(args.saved_path + "/visualization.png")        
    else:
        raise Exception("Error (Please set the 'color_code' flag or 'check_smiles')")
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type = str,
                        default = "./smiles_data/drugbank_smiles.txt", help = "Path of SMILES data for input compounds")
    parser.add_argument("--saved_path", type = str,
                        default = "./output_data", help = "The path where the created 'parmas'(latent variables) is saved")
    parser.add_argument("-check_path", "--check_smiles_path", type = str,
                        default = "./smiles_data/check_smiles.txt", help = "Path of the SMILES data describing the compounds you want to visualize (These compounds must be included in the SMILES data in the original 'smiles_path')") 
    parser.add_argument("--perplexity", type = int,
                        default = 100, help = "perplexity in tSNE")
    parser.add_argument("--n_iter", type = int,
                        default = 1000, help = "n_iter in tSNE")
    parser.add_argument("--random_state", type = int,
                        default = 0, help = "random_state in tSNE")
    parser.add_argument("-color" , "--color_code", action = 'store_true', help = "Whether to color-code according to property information values")
    parser.add_argument("--prop_info", type = str,
                        default = 'nplikeness', help = "Functional information used for learning")
    parser.add_argument("--prepared_path", type = str,
                        default = "./save_data", help = "The path where the created 'input data' is saved (No need to set if you do not use your own property information)")
    args = parser.parse_args()
    print("Now Calculating...")
    main(args)
    print("Finished")