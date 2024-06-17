import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import time
import csv
import torch
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx, to_dgl,from_dgl
from model import GCN,GATv2, SimpleGCN, SGC
from dataloader import *
from trainsplittrain import *



parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--ogmodel', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--remodel', type=str, default='GATv2', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--ogLR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--reLR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters_delete', type=int, default=0, help='maximum number of edge deletions')
parser.add_argument('--max_iters_add', type=int, default=0, help='maximum number of edge additions')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--oghidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--rehidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
args = parser.parse_args()



filename = args.out

max_iterations_deletions = args.max_iters_delete
max_iterations_additions = args.max_iters_add
p = args.dropout


print("Loading dataset...")

if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','DBLP','penn94','reed98']:
    data, num_classes,num_features = load_data(args.dataset)


elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
    data, num_classes,num_features = load_data(args.dataset)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    data, num_classes,num_features = load_data(args.dataset)

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    data, num_classes,num_features = load_data(args.dataset)

else :
    print("Invalid dataset")
    sys.exit()



print()
print("Start Training...")


def create_model(model, num_features, num_classes, hidden_dimension):
    if model == 'GCN':
        return GCN(num_features, num_classes, hidden_dimension, num_layers=args.num_layers)
    elif model == 'GATv2':
        return GATv2(num_features, 8, num_classes)
    elif model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension)
    elif model == 'SGC':
        return SGC(num_features, num_classes)
    else:
        print("Invalid Model")
        sys.exit()





print("Label distiller model...")
ogmodel = create_model(args.ogmodel, num_features, num_classes, args.oghidden_dimension).to(device)
ogoptimizer = torch.optim.Adam(ogmodel.parameters(), lr=args.ogLR, weight_decay=0.0)
ogmodel = ogmodel.to(device)
print(ogmodel)

print("Retraining model...")
remodel = create_model(args.remodel, num_features, num_classes, args.rehidden_dimension).to(device)
reoptimizer = torch.optim.Adam(remodel.parameters(), lr=args.reLR, weight_decay=0.0)
remodel = remodel.to(device)
print(remodel)

gcn_start = time.time()
finaltestaccafter = train_and_get_results(data, ogmodel,ogoptimizer,remodel,reoptimizer,max_iterations_deletions,max_iterations_additions, p)
gcn_end = time.time()

print(f'Final test accuracy after {np.mean(finaltestaccafter):.2f} \u00B1 {np.std(finaltestaccafter):.2f}')


avg_test_acc_after = np.mean(finaltestaccafter)
std_dev_after = np.std(finaltestaccafter)


headers = ['Method','OGModel','Remodel','Dataset','AvgTestAccAfter','DeviationAfter',
        'HiddenDim','LR','Dropout','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow(["GCNGATDeletions",args.ogmodel,args.remodel,args.dataset,avg_test_acc_after,std_dev_after,
                  hidden_dimension, lr, p, gcn_end-gcn_start])
