
This repository contains the code for our AAAI 2025 contributed paper “	
SFR-GNN: Simple and Fast Robust GNNs against Structural Attacks”.

## Basic Environment
* `python >= 3.9` 
* `pytorch >= 1.13.0`
* 
See `requirements.txt` for more information.

## Usage Example
```bash
python main.py --attacker meta --dataset cora
```

## Reminder
The data has already been placed in the `data` folder within the code repository. 
Due to file size upload limitations, the data folder exclusively contains the Cora, Citeseer, and Pubmed datasets.
The modified adjacency matrices should be located in the `ModifiedGraph` folder.
After running the code, the results will be written to a file named **checkpoints_attacker_dataset_perturbation.json** in the `checkpoints` folder.