import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
import methods

planetoid_val_seeds =  [3164711608]
#planetoid_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def train_and_get_results(data, model,p,lr,weight_decay=0):
    trainacclist = []
    avg_testacc = []
    avg_acc_testallsplits = []
    trainallsplits = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss, train_acc


    def val():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


    def test():
            model.eval()
            out= model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc,pred


    for split_idx in range(0,9):
        print(f"Training for index = {split_idx}")
        train_mask = data.train_mask[:,split_idx]
        test_mask = data.test_mask[:,split_idx]
        val_mask = data.val_mask[:,split_idx]

        for seeds in planetoid_val_seeds:
                    set_seed(seeds)
                    print("Start training ....")
                    for epoch in tqdm(range(1, 101)):
                        loss, train_acc = train()
                    val_acc = val()
                    test_acc,pred = test()
                    trainacclist.append(train_acc*100)
                    avg_testacc.append(test_acc*100)
                    print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f} for seed',seeds)
                    print()
        avg_acc_testallsplits.append(np.mean(avg_testacc))
        trainallsplits.append(np.mean(trainacclist))

    avg_acc_testallsplits.append(np.mean(avg_testacc))
    trainallsplits.append(np.mean(trainacclist))

    return avg_acc_testallsplits, trainallsplits


