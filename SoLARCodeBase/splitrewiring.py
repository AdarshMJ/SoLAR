import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
import torch
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx, to_dgl,from_dgl
from torch_geometric.transforms import RandomNodeSplit
from model import GCN,GATv2, SimpleGCN, SGC
import methods
from rewiring import *
from rewiring.spectral_utils import spectral_gap
from dataloader import *
from nodeli import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import pickle
import time
import csv
from trainsplittrain import *
from esnr import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from spectralclustering import *


parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in GATv2')
parser.add_argument('--existing_graph', type=str,default=None, help='.pt file')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters_delete', type=int, default=0, help='maximum number of edge deletions')
parser.add_argument('--max_iters_add', type=int, default=0, help='maximum number of edge additions')
#parser.add_argument('--max_iters', type=int, default=0, help='maximum number of edge change iterations')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
args = parser.parse_args()


# def check_for_leakage(data):
#     train_nodes = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
#     test_nodes = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()

#     # Check if any node in the training set is in the test set
#     leakage_nodes = np.intersect1d(train_nodes, test_nodes)

#     if len(leakage_nodes) > 0:
#         print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets.")
#     else:
        # print("No leakage detected.")



device = torch.device(args.device)
filename = args.out
graphfile = args.existing_graph
max_iterations_deletions = args.max_iters_delete
max_iterations_additions = args.max_iters_add
initialgap = None
fgap = None
data_modifying = None
p = args.dropout
lr = args.LR
hidden_dimension = args.hidden_dimension
avg_testacc = []
avg_acc_testallsplits = []
trainacclist = []
trainallsplits = []


#het_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]

if os.path.exists(graphfile):
  print("Loading graph from .pt file...")
  data = torch.load(graphfile)
  print(data)
  data.y = data.y.squeeze(1)
  if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','CoraFull','DBLP']:
      _, num_classes,num_features = load_data(args.dataset)
  
  elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
        _, num_classes,num_features = load_data(args.dataset)

  
  else:
      num_features = data.num_features
      num_classes = data.num_classes
  #nxgraph = to_networkx(data, to_undirected=True)
  #fgap, _, _, _ = spectral_gap(nxgraph)
  print(f"FinalGap = {fgap}")
  print()

else:
  print("Graph does not exist...")

  if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','DBLP','penn94','reed98']:
      data, num_classes,num_features = load_data(args.dataset)
      print(data)
  elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
      data, num_classes,num_features = load_data(args.dataset)
      #data = to_dgl(data)
      #bidata = data.to_bidirected()
      #data = from_dgl(bidata)
  else :
    path = 'hetdata/'
    filepath = os.path.join( path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index)
    data.y = y
    data.num_classes = num_classes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    num_features = data.num_features
    num_classes = data.num_classes
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Done!..")

datasetname, _ = os.path.splitext(args.dataset)
print()
#nxgraph = to_networkx(data, to_undirected=True)
#print(nxgraph)
#initialgap, _, _, _ = spectral_gap(nxgraph)
print(f"InitialGap = {initialgap}")
print()

##========================= Split the dataset into train/test/val ====================##
#print("Splitting datasets train/val/test...")
#transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test = 0.2,num_val = 0.2)
#data  = transform2(data)
#print(data)
#check_for_leakage(data)
data = data.to(device)
print()
print("Start Training...")

##=========================##=========================##=========================##=========================


def create_model(args, num_features, num_classes, hidden_dimension):
    if args.model == 'GCN':
        return GCN(num_features, num_classes, hidden_dimension, num_layers=args.num_layers)
    elif args.model == 'GATv2':
        return GATv2(num_features, 8, num_classes)
    elif args.model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension)
    elif args.model == 'SGC':
        return SGC(num_features, num_classes)
    else:
        print("Invalid Model")
        sys.exit()


OGmodel = GCN(num_features, hidden_dimension,num_classes)
OGmodel.to(device)
print(OGmodel)
remodel = GCN(num_features, hidden_dimension,num_classes)
remodel.to(device)
gcn_start = time.time()
finaltestaccafter = train_and_get_results(data, OGmodel,remodel,max_iterations_deletions,max_iterations_additions, p, lr)
gcn_end = time.time()

print(f'Final test accuracy after {np.mean(finaltestaccafter):.2f} \u00B1 {np.std(finaltestaccafter):.2f}')


avg_test_acc_after = np.mean(finaltestaccafter)
std_dev_after = np.std(finaltestaccafter)


headers = ['Method','Model','Dataset','AvgTestAccAfter','DeviationAfter',
        'HiddenDim','LR','Dropout','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow(["GCNDeletions",args.model,args.dataset,avg_test_acc_after,std_dev_after,
                  hidden_dimension, lr, p, gcn_end-gcn_start])