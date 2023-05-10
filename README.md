# Deep generative model of constructing chemical latent space for macromolecular structures
## Overview
The structural diversity of chemical libraries, which are systematic collections of compounds that have potential to bind to biomolecules, can be represented by chemical latent space. \
In this study, we developed a new deep-learning method, called NP-VAE, based on variational autoencoder for handling natural compounds. NP-VAE enables generation of a chemical latent space that project large and complex compound structures including chirality.
## Requirements
* python 3.7.6 (with following packages)
  * dgl 0.6.1
  * matplotlib 3.1.3
  * mkl_service 2.3.0
  * numpy 1.18.1
  * rdkit 2020.03.3.0
  * scikit_learn 0.22.1
  * torch 1.7.0
## 
## Usage
Please select and execute the following python files according to your purpose.
* `preprocessing.py` performs preprocessing. Please run it first when you want to train this model.
* `train.py` is executed when you want to train this model on your own data set and obtain new parameters.
* `calculate_z.py` calculates latent variables corresponding to input compounds based on learned parameters.
* `evaluate.py` is executed to evaluate the reconstruction accuracy after training.
* `visualize.py` visualizes latent variables by dimensionality reduction. You can also colorize and see the distribution of specific compounds you set.
* `generate.py` generates new compound structures from the space around a specified compound based on a learned model.
  
The saved data from our training on the DrugBank&Project dataset can be found in the `pre-train` folder. If you want to use it in the following process, please specify this path.
### Train the model
The trained parameters are published, but if you want to train the model on your own dataset, run `preprocessing.py` and `train.py`.\
Some of the main parameters to be set and command examples are shown below.
#### 1. Preprocessing
```
python preprocessing.py --smiles_path ./smiles_data/hoge.txt --save_path ./save_data
```
* --smiles_path, `SMILES_PATH`: Please specify the path of your SMILES data to be used for training.
* --save_path, `SAVE_PATH`:Please specify the path to save your obtained data. **Under this path, please make a new folder named `input_data`. Then, under `input_data`, please make a new folder named `weights`.**
#### 2. Train
The training uses multiple GPUs for acceleration. please make sure your GPUs are available.
```
python train.py --smiles_path ./smiles_data/hoge.txt --prepared_path ./save_data --save_path ./param_data
```
* --prepared_path, `PREPARED_PATH`: The path where the created "input data" folder is saved. **This path is the same as `SAVE_PATH` in the preprocessing.**
* --save_path, `SAVE_PATH`: Please specify the path to save parameters of this model after training.
* --max_epoch, `MAX_EPOCH`: Number of epochs. default:`100`
* --save_epoch, `SAVE_EPOCH`: Epoch intervals to save parameters. default:`20`
* --prop_info, `PROP_INFO`: Property information used for learning. **If you want to specify a property other than `nplikeness` or `logp`, please name the pickle file in which you saved the property `prop_info` and save it in the `input_data` folder.** default:`nplikeness`
### Calculate latent variables
You can calculate latent variables corresponding to your input compounds based on learned parameters.\
**If you want to obtain latent variables that match your compound structure based on published parameters without training, please run `preprocessing.py` first to complete the preprocessing.**
```
python calculate_z.py --smiles_path ./smiles_data/hoge.txt --prepared_path ./save_data --load_path ./params_data --save_path ./output_data
```
* --smiles_path, `SMILES_PATH`: Same as below.
* --prepared_path, `PREPARED_PATH`: Same as below.
* --load_path, `LOAD_PATH`: The path where learned parameters are saved. 
* --save_path, `SAVE_PATH`: Path to save latent variables and other output values.
* --load_epoch, `LOAD_EPOCH`: Epoch specified for loading parameters. default:`100`
### Evaluate reconstruction accuracy
This process does not need to be performed, but it helps to verify the fitting accuracy of the model.
```
python evaluate.py --smiles_path ./smiles_data/hoge.txt --saved_path ./output_data
```
* --saved_path, `SAVED_PATH`: This path is the same as `SAVE_PATH` in the calculation of the latent variables.
* --check3D_result, -check3D: This code returns the consistency of three-dimensional structures when this flag is set (Longer time required for calculations).
### Visualize latent variables
The acquired latent variables are dimensionally reduced by tSNE and the visualization results can be obtained as a png file. The png file is saved under the `saved_path` you specified.\
If you want to colorize and see the distribution of specific compounds, please prepare a txt file describing them separately.\
If you just want to see the appearance by color coding for each property information value, there is no need to prepare a separate txt file. In that case, please set the `-color` flag.
```
python visualize.py --smiles_path ./smiles_data/hoge.txt --saved_path ./output_data -check_path ./smiles_data/target_smiles.txt -color
```
* -check_path, --check_smiles_path `CHECK_SMILES_PATH`: Path of the SMILES data describing the compounds you want to visualize (These compounds must be included in the SMILES data in the original `smiles_path`)
* -color, --color_code: Whether to color-code according to property information values.
### Generate new compound structures
You can generate new compound structures from the space around a compound you specified. A SDF file is saved under the `saved_path` you specified.
```
python generate.py --smiles_path ./smiles_data/hoge.txt --prepared_path ./save_data --load_path ./params_data --saved_path ./output_data -target `c1ccccc1`
```
* -target, --target_smiles, `TARGET_SMILES`: SMILES of the target point compound. (The target SMILES must be included in the SMILES data in the `smiles_path`)
* -ngen, --num_gen_mols, `NUM_GEN_MOLS`: Number of new compounds generated before refining. default:`10000`
* -nmol, --num_new_mols, `NUM_NEW_MOLS`: Number of new compounds generated after refining. default:`5000`
* -r, --search_radius, `SEARCH_RADIUS`: Search radius from the target compound. default:`1`