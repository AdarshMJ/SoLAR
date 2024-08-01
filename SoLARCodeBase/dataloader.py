import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork, Coauthor, EmailEUCore,Amazon,HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_data(datasetname):
        path = '../data/' + datasetname
        if datasetname in ['Cora','Citeseer','Pubmed']:
            dataset = Planetoid(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            data = data.to(device)
            print(data)
            #print(data)

        elif datasetname in ['CS','Physics']:
            dataset = Coauthor(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            data = data.to(device)

            print(data)
            
        elif datasetname in ['Computers','Photo']:
            dataset = Amazon(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            data = data.to(device)
            print(data)


        elif datasetname in ['Roman-empire','Minesweeper']:
            dataset = HeterophilousGraphDataset(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            print()
            data = transform(dataset[0])
            data = data.to(device)
            print(data)


        elif datasetname in ['cornell.npz', 'texas.npz','wisconsin.npz']:
            path = '/home/data/'
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
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.2,num_val=0.2)
            data  = transform2(data)
            data = data.to(device)
            print(data)
            num_features = data.num_features
            num_classes = data.num_classes
            print("Done!..")



        elif datasetname in ['chameleon_filtered.npz', 'squirrel_filtered.npz','actor.npz']:
            path = '/home/data/'
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
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            print()
            data = data.to(device)
            print(data)
            num_features = data.num_features
            num_classes = data.num_classes
            print("Done!..")

        else:
            raise ValueError(f"Dataset {datasetname} not found")

        return data, num_classes,num_features


##Sanity check if data leakage occurs when we split data###
# for i in range(100):
#     train_nodes = data.train_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     test_nodes = data.test_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     val_nodes = data.val_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     leakage_nodes = np.intersect1d(train_nodes, test_nodes)
#     if len(leakage_nodes) > 0:
#         print(
#             f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets."
#         )
#     else:
#         print("No leakage detected.")
