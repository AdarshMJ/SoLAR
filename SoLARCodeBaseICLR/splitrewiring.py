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
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from spectralclustering import *

parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--ogmodel', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--remodel', type=str, default='SGC', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--oglr', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--relr', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters_delete', type=int, default=0, help='maximum number of edge deletions')
parser.add_argument('--max_iters_add', type=int, default=0, help='maximum number of edge additions')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--oghidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--rehidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--num_train',type=int,default='20',help='Number of training nodes per class')
parser.add_argument('--num_val',type=int,default='500',help='Number of validation nodes')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
args = parser.parse_args()



filename = args.out

max_iterations_deletions = args.max_iters_delete
max_iterations_additions = args.max_iters_add
p = args.dropout
oghidden_dimension = args.oghidden_dimension
rehidden_dimension = args.rehidden_dimension
oglr = args.oglr
relr = args.relr


print("Loading dataset...")

if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','DBLP','penn94','reed98']:
    data, num_classes,num_features,num_train_nodes,num_test_nodes,num_val_nodes = load_data(args.dataset, args.num_train, args.num_val)



elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
    data, num_classes,num_features = load_data(args.dataset,args.num_train,args.num_val)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print()
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
    data  = transform2(data)
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Splitting datasets train/val/test...")
    # transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_train_per_class = 10,num_val=20)
    # #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.2,num_val=0.2)
    # data  = transform2(data)
    transform2 = RandomNodeSplit(split="test_rest", num_splits=100, num_train_per_class=args.num_train, num_val=args.num_val)
    data = transform2(data)
    #data.train_mask = data.train_mask | data.val_mask
    #data.val_mask = torch.zeros_like(data.val_mask)  # Clear validation mask
    num_train_nodes = data.train_mask.sum().item()
    num_val_nodes = data.val_mask.sum().item()
    num_test_nodes = data.test_mask.sum().item()
    print()
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

else :
    print("Invalid dataset")
    sys.exit()



print()
print(f"Number of training nodes: {num_train_nodes/100}")
print(f"Number of validation nodes: {num_val_nodes/100}")
print(f"Number of test nodes: {num_test_nodes/100}")
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

ogmodel = ogmodel.to(device)
print(ogmodel)

print("Retraining model...")
remodel = create_model(args.remodel, num_features, num_classes, args.rehidden_dimension).to(device)

remodel = remodel.to(device)
print(remodel)

gcn_start = time.time()
finaltestaccafter,finalvalaccafter,ActualEdgesRemoved, ActualEdgesAdded = train_and_get_results(data, ogmodel, oglr, remodel, relr, max_iterations_deletions, max_iterations_additions)
gcn_end = time.time()


avg_test_acc_after = np.mean(finaltestaccafter)
sample_size = len(finaltestaccafter)
std_dev_after = 2 * np.std(finaltestaccafter)/(np.sqrt(sample_size))

avg_val_acc_after = np.mean(finalvalaccafter)


print(f'Final test accuracy after {(avg_test_acc_after):.2f} \u00B1 {(std_dev_after):.2f}')
print(f'Final validation accuracy after {(avg_val_acc_after):.2f}')


headers = ['Method','Dataset','AvgTestAccAfter','DeviationAfter','AvgValAccAfter',
        'OGHiddenDim','ReHiddenDim','ogLR','reLR','TrainNodes','TestNodes','ValNodes','EdgesRemoved','EdgesAdded','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(headers)
    
    if args.max_iters_delete > 0 and args.max_iters_add > 0:
        operation = 'deladd'
    elif args.max_iters_delete > 0:
        operation = 'del'
    elif args.max_iters_add > 0:
        operation = 'add'
    else:
        operation = 'Baseline'

    method = f"{args.ogmodel}{args.remodel}{operation}"

    writer.writerow([method, args.dataset, f"{(avg_test_acc_after):.4f}", f"{(std_dev_after):.4f}",f"{(avg_val_acc_after):.4f}",
                     oghidden_dimension, rehidden_dimension, oglr, relr,num_train_nodes/100,num_test_nodes/100,num_val_nodes/100,ActualEdgesRemoved,ActualEdgesAdded, gcn_end - gcn_start])
