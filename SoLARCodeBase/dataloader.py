import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork, Coauthor, EmailEUCore,Amazon,HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents

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
            #print(data)

        elif datasetname in ['CS','Physics']:
            dataset = Coauthor(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print(data)
            
        elif datasetname in ['Computers','Photo']:
            dataset = Amazon(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print(data)


        elif datasetname in ['Roman-empire','Minesweeper']:
            dataset = HeterophilousGraphDataset(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print(data)

        else:
            raise ValueError(f"Dataset {datasetname} not found")

        return data, num_classes,num_features