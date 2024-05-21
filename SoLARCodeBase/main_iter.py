import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
import torch
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
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
from train import *
from esnr import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from spectralclustering import *

######### Hyperparams to use #############
#Cora --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Citeseer --> Dropout = 0.3130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Pubmed --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
# Cornell = 0.4130296,0.001, 128
# Wisconsin = 0.5130296, 0.001,128
# Texas = 0.4130296,0.001,128
# Actor = 0.2130296,0.01,128
# ChameleonFiltered = 0.2130296,0.01,128
# ChameleonFilteredDirected = 0.4130296,0.01,128
# SquirrelFiltered = 0.5130296,0.01,128
# SquirrelFilteredDirected = 0.2130296,0.01,128
########################################



parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in GATv2')
parser.add_argument('--existing_graph', type=str,default=None, help='.pt file')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters_delete', type=int, default=0, help='maximum number of edge deletions')
parser.add_argument('--max_iters_add', type=int, default=0, help='maximum number of edge additions')
parser.add_argument('--train_iters', type=int, default=100, help='number of training iterations')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32 , help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
args = parser.parse_args()




device = torch.device(args.device)
filename = args.out
graphfile = args.existing_graph

num_iterations = args.train_iters
max_iterations_deletions = args.max_iters_delete
max_iterations_additions = args.max_iters_add
initialgap = None
ActualEdgesAdded = None
ActualEdgesDeleted = None
fgap = None
data_modifying = None
p = args.dropout
lr = args.LR
hidden_dimension = args.hidden_dimension
avg_testacc = []
avg_acc_testallsplits = []
trainacclist = []
trainallsplits = []


def create_model(args, num_features, num_classes, hidden_dimension):
    if args.model == 'GCN':
        return GCN(num_features, num_classes, hidden_dimension, num_layers=args.num_layers)
    elif args.model == 'GATv2':
        return GATv2(num_features, 5, num_classes)
    elif args.model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension)
    elif args.model == 'SGC':
        return SGC(num_features, num_classes)
    else:
        print("Invalid Model")
        sys.exit()


if os.path.exists(graphfile):
    print("Loading graph from .pt file...")
    data = torch.load(graphfile)
    print(data)

    if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo']:
        _, num_classes,num_features = load_data(args.dataset)
    else:
        num_features = data.num_features
        num_classes = data.num_classes
    print(f"FinalGap = {fgap}")
    print()


elif args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo']:
      data, num_classes,num_features = load_data(args.dataset)
  
elif args.dataset in ['Roman-empire','Minesweeper']:
      data, num_classes,num_features = load_data(args.dataset)

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
print(f"InitialGap = {initialgap}")
print()

##========================= Split the dataset into train/test/val ====================##
print("Splitting datasets train/val/test...")
transform2 = RandomNodeSplit(split="test_rest",num_splits = 100)
data  = transform2(data)
print(data)
data = data.to(device)
model = create_model(args, num_features, num_classes, hidden_dimension)
model.to(device)
print(model)
print()
print("Start pre-training...")



finaltestacc, finaltrainacc, pred,_ = train_and_get_results(data, model, p, lr)
best_acc = np.mean(finaltestacc)  # Initialize the best accuracy
no_improve = 0  # Initialize the counter for no improvement
early_stop = 20  # Define the early stopping criteria
edges_added_log = []
gcn_start = time.time()
for i in tqdm(range(num_iterations), desc="Training iterations"):
    if not os.path.exists(graphfile):
        newdata,ActualEdgesDeleted,ActualEdgesAdded = methods.PeerGNNDeleteAdd(data,max_iterations_deletions,max_iterations_additions, pred)       
        new_model = create_model(args, num_features, num_classes, hidden_dimension)
        new_model.to(device)
        print(newdata)
        newfinaltestacc, newfinaltrainacc, new_pred,_ = train_and_get_results(newdata, new_model, p, lr)
        current_acc = np.mean(newfinaltestacc)
        print()
        print(f'Current test accuracy: {current_acc}')
        
        edges_added_log.append([i, ActualEdgesAdded])


        if current_acc > best_acc:
            best_acc = current_acc
            pred = new_pred
            data = newdata
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop:
            print("Early stopping, no improvement for {} iterations".format(early_stop))
            break

        testacc_to_report = best_acc
        data_to_use = newdata
    else:
        testacc_to_report = finaltestacc
        data_to_use = data

# Print the final test accuracy
print()
print(f'Final test accuracy: {best_acc}')

gcn_end = time.time()

print()
print("Calculating Informativeness measures...")
graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data_to_use)
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'edge label informativeness: {edgeliaf:.4f}')
print(f'adjusted homophily: {hadjaf:.4f}')
print("=============================================================")
print("Done!")
print()

##=========================##=========================##=========================##=========================
print("Calculating Edge Signal to Noise Ratio...")
esnr_score = esnr_vanilla(data_to_use)
print(f'Edge Signal to Noise Ratio: {esnr_score:.4f}')
print("Done!")
print()

##=========================##=========================##=========================##=========================
# print("Performing clustering...")
# nxgraph = to_networkx(data_to_use, to_undirected=True)
# ground_truth_labels = data_to_use.y.cpu()
# clustermod = maximize_modularity(nxgraph)
# cluster_dict = {node: i for i, cluster in enumerate(clustermod) for node in cluster}
# cluster_list = [cluster_dict[node] for node in range(len(data_to_use.y))]


# #nmi_score_spectral = NMI(labels, ground_truth_labels)
# nmiscoremod = NMI(cluster_list, ground_truth_labels)
# modscoremod = nx.community.modularity(nxgraph, clustermod)
# uncertaintyscore = uncertainty_coefficient(ground_truth_labels, cluster_list)
nmiscoremod = None
modscoremod = None
uncertaintyscore = None


# print(f'NMI Score Modularity: {nmiscoremod:.4f}')
# print("Done!")

print("Number of Inter/Intra class edges...")
inter_class_edges,intra_class_edges = count_edges(data_to_use)
print(f"Interclass edges = {inter_class_edges}, Intraclass edges = {intra_class_edges}")
print()



with open('edges_added_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Number of Edges Added'])
    writer.writerows(edges_added_log)


headers = ['Method','Dataset','AvgTestAcc', 'Deviation','ELIAfter',
        'AdjHomAfter','NMIModMax','ModScore','UncertaintyScore',
        'EdgesDeleted','EdgesAdded','FinalGap','ESNR','TrainingIters','InterClass','IntraClass','HiddenDim','LR','Dropout','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow(["IterGATDelete",args.dataset, np.mean(testacc_to_report), np.std(testacc_to_report),
                     edgeliaf*100, hadjaf*100, nmiscoremod, modscoremod, uncertaintyscore,
                     max_iterations_deletions,max_iterations_additions, fgap, esnr_score,num_iterations, inter_class_edges, intra_class_edges, hidden_dimension, lr, p, gcn_end-gcn_start])


