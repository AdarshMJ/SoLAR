import time
import torch
import networkx as nx
from dataloader import *
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.special import comb
from rewiring.fastrewiringKupdates import *
from rewiring.MinGapKupdates import *
from rewiring.fosr import *
from rewiring.spectral_utils import *
from rewiring.sdrf import *
from torch_geometric.utils import to_networkx,from_networkx,homophily
import torch.nn.functional as F
from torch_geometric.data import Data
import csv
from scipy.stats import entropy
import copy


def proxydelmin(data, nxgraph, max_iterations):
    print("Deleting edges to minimize the gap...")
    start_algo = time.time()
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_delete_min, "proxydeletemin", max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = (end_algo - start_algo)
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    print(newdata)
    data.edge_index = torch.cat([newdata.edge_index]) 
    return data,fgap,data_modifying

def proxydelmax(data, nxgraph, max_iterations):
      print("Deleting edges to maximize the gap...")
      start_algo = time.time()
      newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydeletemax",max_iter=max_iterations,updating_period=1)
      newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
      end_algo = time.time()
      fgap,_, _, _ = spectral_gap(newgraph)
      print()
      print(f"FinalGap = {fgap}") 
      data_modifying = (end_algo - start_algo)
      print(data_modifying)  
      newdata = from_networkx(newgraph)
      print(newdata)
      data.edge_index = torch.cat([newdata.edge_index])  
      return data,fgap,data_modifying

def proxyaddmax(data, nxgraph, max_iterations):
        print("Adding edges to maximize the gap...")
        start_algo = time.time()
        newgraph = process_and_update_edges(nxgraph, rank_by_proxy_add, "proxyaddmax",max_iter=max_iterations,updating_period=1)
        newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
        end_algo = time.time()
        fgap,_, _, _ = spectral_gap(newgraph)
        print()
        print(f"FinalGap = {fgap}") 
        data_modifying = (end_algo - start_algo)
        print(data_modifying)  
        newdata = from_networkx(newgraph)
        print(newdata)
        data.edge_index = torch.cat([newdata.edge_index])  
        return data,fgap,data_modifying


def proxyaddmin(data, nxgraph, max_iterations):
        print("Adding edges to minimize the gap...")
        start_algo = time.time()
        newgraph = min_and_update_edges(nxgraph, rank_by_proxy_add_min, "proxyaddmin",max_iter=max_iterations,updating_period=1)
        newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
        end_algo = time.time()
        fgap,_, _, _ = spectral_gap(newgraph)
        print()
        print(f"FinalGap = {fgap}") 
        data_modifying = (end_algo - start_algo)
        print(data_modifying)  
        newdata = from_networkx(newgraph)
        print(newdata)
        data.edge_index = torch.cat([newdata.edge_index])  
        return data,fgap,data_modifying


def fosr(data, max_iterations):
      print("Adding edges using FoSR...")
      start_algo = time.time()
      for j in tqdm(range((max_iterations))):
        edge_index,edge_type,_,prod = edge_rewire(data.edge_index.numpy(), num_iterations=1)      
        data.edge_index = torch.tensor(edge_index)
      data.edge_index = torch.cat([data.edge_index])
      end_algo = time.time()
      data_modifying = (end_algo - start_algo)
      newgraph = to_networkx(data, to_undirected=True)
      fgap,_, _, _ = spectral_gap(newgraph)
      return data, fgap, data_modifying

def sdrf(data, max_iterations,removal_bound,tau):
          #print("Rewiring using SDRF...")
          start_algo = time.time()
          Newdatapyg = sdrf(data,max_iterations,removal_bound,tau)
          end_algo = time.time()
          data_modifying = (end_algo - start_algo)
          newgraph = to_networkx(Newdatapyg, to_undirected=True)
          fgap,_, _, _ = spectral_gap(newgraph)
          data = from_networkx(Newdatapyg)
          return data, fgap, data_modifying


def remove_random_edges(data, nxgraph,max_iter):
    print("Deleting edges randomly to minimize the gap...")
    all_edges = list(nxgraph.edges())
    
    # Check if the graph has enough edges to remove
    if len(all_edges) < num_edges_to_remove:
        raise ValueError("The graph has fewer edges than the number of edges to remove")

    # Randomly select edges to remove
    edges_to_remove = random.sample(all_edges, num_edges_to_remove)

    start_algo = time.time()
    for edge in edges_to_remove:
        nxgraph.remove_edge(*edge)
    end_algo = time.time()
    fgap,_, _, _ = spectral_gap(nxgraph)
    data_modifying = (end_algo - start_algo)
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    data.edge_index = torch.cat([newdata.edge_index])  
    return data,fgap,data_modifying


# def PeerGNNDelete(data, num_edges_to_remove, pred):
#     inter_class_edges = 0
#     intra_class_edges = 0

#     edge_index = data.edge_index.cpu().numpy()
#     labels = pred.cpu().numpy()  # Use predicted labels

#     removed_edges = []

#     for edge in edge_index.T:
#         node1_class = labels[edge[0]]
#         node2_class = labels[edge[1]]

#         if node1_class == node2_class:
#             intra_class_edges += 1
#         else:
#             inter_class_edges += 1

#     print(f"Number of inter-class edges that can be deleted: {inter_class_edges}")

#     # Check if the number of edges to remove is less than the total number of inter-class edges
#     if num_edges_to_remove >= inter_class_edges:
#         raise ValueError("Cannot remove all inter-class edges. Adjust the number of edges to remove.")

#     inter_class_edges = 0  # Reset the count

#     for edge in edge_index.T:
#         node1_class = labels[edge[0]]
#         node2_class = labels[edge[1]]

#         if node1_class != node2_class and inter_class_edges < num_edges_to_remove:
#             removed_edges.append(edge)
#             inter_class_edges += 1

#     for edge in removed_edges:
#         edge_index = np.delete(edge_index, np.where((edge_index.T == edge).all(axis=1)), axis=1)

#     data.edge_index = torch.from_numpy(edge_index).to(data.edge_index.device)

#     return data


# def PeerGNNDeleteAdd(data, num_edges_to_remove, num_edges_to_add, pred):
#     inter_class_edges = 0
#     intra_class_edges = 0

#     edge_index = data.edge_index.cpu().numpy()
#     labels = pred.cpu().numpy()  # Use predicted labels

#     removed_edges = []
#     added_edges = []

#     for edge in edge_index.T:
#         node1_class = labels[edge[0]]
#         node2_class = labels[edge[1]]

#         if node1_class == node2_class:
#             intra_class_edges += 1
#         else:
#             inter_class_edges += 1

#     print(f"Number of inter-class edges that can be deleted: {inter_class_edges}")

#     # Check if the number of edges to remove is less than the total number of inter-class edges
#     if num_edges_to_remove >= inter_class_edges:
#         raise ValueError("Cannot remove all inter-class edges. Adjust the number of edges to remove.")

#     inter_class_edges = 0  # Reset the count

#     for edge in edge_index.T:
#         node1_class = labels[edge[0]]
#         node2_class = labels[edge[1]]

#         if node1_class != node2_class and inter_class_edges < num_edges_to_remove:
#             removed_edges.append(edge)
#             inter_class_edges += 1

#     for edge in removed_edges:
#         edge_index = np.delete(edge_index, np.where((edge_index.T == edge).all(axis=1)), axis=1)

#     # Add new intra-class edges
#     for _ in range(num_edges_to_add):
#         # Randomly select two nodes
#         node1, node2 = np.random.choice(labels.shape[0], 2, replace=False)

#         # Check if the nodes are of the same class and not already connected
#         if labels[node1] == labels[node2] and not np.any((edge_index.T == [node1, node2]).all(axis=1)):
#             added_edges.append([node1, node2])

#     for edge in added_edges:
#         edge_index = np.append(edge_index, np.transpose([edge]), axis=1)

#     data.edge_index = torch.from_numpy(edge_index).to(data.edge_index.device)

#     return data

def kd_retention(model, data: Data, noise_level: float):
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out_teacher = model(data.x, data.edge_index)
        data_teacher = F.softmax(out_teacher, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_teacher])
        feats_noise = copy.deepcopy(data.x)
        feats_noise += torch.randn_like(feats_noise) * noise_level
        data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
    with torch.no_grad():
        out_noise = model(data_noise.x, data_noise.edge_index)
        out_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
        weight_s = np.abs(np.array([entropy(on) for on in out_noise]) - weight_t)
        delta_entropy = weight_s / np.max(weight_s)
    return delta_entropy

def rank_edges_by_entropy(model, data: Data, noise_level: float):
    delta_entropy = kd_retention(model, data, noise_level)
    edge_index = data.edge_index.cpu().numpy()
    edge_entropies = []
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]
        avg_entropy = (delta_entropy[node1] + delta_entropy[node2]) / 2
        edge_entropies.append((node1, node2, avg_entropy))
    # Sort edges by average entropy
    edge_entropies.sort(key=lambda x: x[2], reverse=True)
    return edge_entropies

# def write_edges_to_csv(model, data: Data, noise_level: float, output_file: str, train_mask, val_mask, test_mask):
#     edge_entropies = rank_edges_by_entropy(model, data, noise_level)
    
#     # Convert masks to numpy arrays if they're not already
#     train_mask = train_mask.cpu().numpy() if isinstance(train_mask, torch.Tensor) else train_mask
#     val_mask = val_mask.cpu().numpy() if isinstance(val_mask, torch.Tensor) else val_mask
#     test_mask = test_mask.cpu().numpy() if isinstance(test_mask, torch.Tensor) else test_mask

#     # Ensure masks are 2D
#     train_mask = np.atleast_2d(train_mask)
#     val_mask = np.atleast_2d(val_mask)
#     test_mask = np.atleast_2d(test_mask)

#     # Create a function to determine the split for a node
#     def get_node_split(node):
#         if train_mask[node].any():
#             return 'train'
#         elif val_mask[node].any():
#             return 'val'
#         elif test_mask[node].any():
#             return 'test'
#         else:
#             return 'unknown'

#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Node1', 'Node2', 'Entropy', 'Node1_Split', 'Node2_Split'])
        
#         for node1, node2, entropy in edge_entropies:
#             writer.writerow([
#                 node1, 
#                 node2, 
#                 entropy, 
#                 get_node_split(node1), 
#                 get_node_split(node2)
#             ])

# def PeerGNNDeleteAdd(model, data, num_edges_to_remove, num_edges_to_add, pred, train_mask, noise_level):
#     np.random.seed(42)

#     edge_index = data.edge_index.cpu().numpy()
#     pred = pred.cpu().numpy()  # Use predicted labels
#     labels = data.y.cpu().numpy()  # Ground truth labels
#     train_mask = train_mask.cpu().numpy()

#     # Create a combined label array
#     combined_labels = np.where(train_mask, labels, pred)

#     removed_edges = []
#     added_edges = []

#     # Get ranked edges by entropy
#     ranked_edges = rank_edges_by_entropy(model, data, noise_level)

#     inter_class_edges = 0
#     intra_class_edges = 0

#     for edge in ranked_edges:
#         node1, node2, _ = edge
#         node1_class = combined_labels[node1]
#         node2_class = combined_labels[node2]

#         if node1_class != node2_class:
#             inter_class_edges += 1
#         else:
#             intra_class_edges += 1

#     print()
#     print(f"Number of inter-class edges that can be deleted: {inter_class_edges}")
#     print(f"Number of intra-class edges that can be added: {intra_class_edges}")
#     print()

#     # Check if the number of edges to remove is less than the total number of inter-class edges
#     if num_edges_to_remove > inter_class_edges:
#         print("The number of edges to remove exceeds the number of inter-class edges. Removing the maximum possible number of inter-class edges.")
#         num_edges_to_remove = inter_class_edges

#     if num_edges_to_add > intra_class_edges:
#         print("The number of edges to add exceeds the number of possible intra-class edges. Adding the maximum possible number of intra-class edges.")
#         num_edges_to_add = intra_class_edges

#     inter_class_edges = 0  # Reset the count

#     for edge in ranked_edges:
#         node1, node2, _ = edge
#         node1_class = combined_labels[node1]
#         node2_class = combined_labels[node2]

#         if node1_class != node2_class and inter_class_edges < num_edges_to_remove:
#             removed_edges.append([node1, node2])
#             inter_class_edges += 1

#     for edge in removed_edges:
#         edge_index = np.delete(edge_index, np.where((edge_index.T == edge).all(axis=1)), axis=1)
        
#     print(f"Number of edges requested to remove: {num_edges_to_remove}")
#     print(f"Number of edges actually removed: {len(removed_edges)}")    

#     for _ in range(num_edges_to_add):
#         # Randomly select two nodes
#         node1, node2 = np.random.choice(combined_labels.shape[0], 2, replace=False)

#         # Check if the nodes are of the same class and not already connected
#         if combined_labels[node1] == combined_labels[node2] and not np.any((edge_index.T == [node1, node2]).all(axis=1)):
#             added_edges.append([node1, node2])

#     for edge in added_edges:
#         edge_index = np.append(edge_index, np.transpose([edge]), axis=1)

#     data.edge_index = torch.from_numpy(edge_index).to(data.edge_index.device)
#     print(f"Number of edges requested to add: {num_edges_to_add}")
#     print(f"Number of edges actually added: {len(added_edges)}")
#     print()
#     return data, len(removed_edges), len(added_edges)




def PeerGNNDeleteAdd(data, num_edges_to_remove, num_edges_to_add, pred, train_mask,val_mask):
    np.random.seed(42)
    inter_class_edges = 0
    intra_class_edges = 0

    edge_index = data.edge_index.cpu().numpy()
    pred = pred.cpu().numpy()  # Use predicted labels
    labels = data.y.cpu().numpy()  # Ground truth labels
    train_mask = train_mask.cpu().numpy()
    val_mask = val_mask.cpu().numpy()

    # Create a combined label array
    combined_labels = np.where(train_mask|val_mask, labels, pred)
    #combined_labels = np.where(train_mask, labels, pred)
    removed_edges = []
    added_edges = []

    # Count the number of nodes in each class
    class_counts = np.bincount(combined_labels)

    # Calculate the maximum possible number of intra-class edges
    max_intra_class_edges = sum(comb(count, 2) for count in class_counts)

    for edge in edge_index.T:
        node1_class = combined_labels[edge[0]]
        node2_class = combined_labels[edge[1]]

        if node1_class == node2_class:
            intra_class_edges += 1
        else:
            inter_class_edges += 1
    print()
    print(f"Number of inter-class edges that can be deleted: {inter_class_edges}")
    print(f"Number of intra-class edges that can be added: {max_intra_class_edges - intra_class_edges}")
    print()
    # Check if the number of edges to remove is less than the total number of inter-class edges
    if num_edges_to_remove > inter_class_edges:
        print("The number of edges to remove exceeds the number of inter-class edges. Removing the maximum possible number of inter-class edges.")
        num_edges_to_remove = inter_class_edges-20

    if num_edges_to_add > max_intra_class_edges - intra_class_edges:
        print("The number of edges to add exceeds the number of possible intra-class edges. Adding the maximum possible number of intra-class edges.")
        num_edges_to_add = max_intra_class_edges - intra_class_edges
        
    inter_class_edges = 0  # Reset the count

    for edge in edge_index.T:
        node1_class = combined_labels[edge[0]]
        node2_class = combined_labels[edge[1]]

        if node1_class != node2_class and inter_class_edges < num_edges_to_remove:
            removed_edges.append(edge)
            inter_class_edges += 1

    for edge in removed_edges:
        edge_index = np.delete(edge_index, np.where((edge_index.T == edge).all(axis=1)), axis=1)
        
    print(f"Number of edges requested to remove: {num_edges_to_remove}")
    print(f"Number of edges actually removed: {len(removed_edges)}")    
    for _ in range(num_edges_to_add):
        # Randomly select two nodes
        node1, node2 = np.random.choice(combined_labels.shape[0], 2, replace=False)

        # Check if the nodes are of the same class and not already connected
        if combined_labels[node1] == combined_labels[node2] and not np.any((edge_index.T == [node1, node2]).all(axis=1)):
            added_edges.append([node1, node2])

    for edge in added_edges:
        edge_index = np.append(edge_index, np.transpose([edge]), axis=1)

    data.edge_index = torch.from_numpy(edge_index).to(data.edge_index.device)
    print(f"Number of edges requested to add: {num_edges_to_add}")
    print(f"Number of edges actually added: {len(added_edges)}")
    print()
    return data, len(removed_edges), len(added_edges)