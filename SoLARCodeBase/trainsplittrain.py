# I need to call this file from the main.py file to train the model and get the results
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
import methods
import copy

#planetoid_val_seeds =  [3164711608]
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



set_seed(3164711608)
def train_and_get_results(data, ogmodel,ogoptimizer,remodel,reoptimizer,edgesdelete,edgesadd,p):
    avg_testacc_before = []
    avg_acc_testallsplits_before = []
    avg_testacc_after = []
    avg_acc_testallsplits_after = []


    criterion = torch.nn.CrossEntropyLoss()

    def train(model,optimizer):
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss


    def val(model):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


    def test(model):
            model.eval()
            out= model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc,pred

    original_data = copy.deepcopy(data)
    
    for split_idx in range(0,100):
        data = copy.deepcopy(original_data)
        train_mask = data.train_mask[:,split_idx]
        test_mask = data.test_mask[:,split_idx]
        val_mask = data.val_mask[:,split_idx]
        print(f"Training for index = {split_idx}")
        for epoch in tqdm(range(1, 101)):
            loss = train(ogmodel,ogoptimizer)
        val_acc = val(ogmodel)
        test_acc,pred = test(ogmodel)
        avg_testacc_before.append(test_acc*100)
        print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f} for seed')
        print()
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))
        
        print()
        print(f"Rewiring for index = {split_idx} -- Deleting {edgesdelete} and Adding {edgesadd} edges")
        data, ActualEdgesRemoved, ActualEdgesAdded = methods.PeerGNNDeleteAdd(data, edgesdelete, edgesadd, pred,train_mask)
        print(data)
        print()
        print("Start re-training ....")
        for epoch in tqdm(range(1, 101)):
            loss = train(remodel,reoptimizer)
        val_acc = val(remodel)
        test_acc,pred= test(remodel)
        avg_testacc_after.append(test_acc*100)
        print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f}')
        print()
        avg_acc_testallsplits_after.append(np.mean(avg_testacc_after))
    
    return avg_acc_testallsplits_after


