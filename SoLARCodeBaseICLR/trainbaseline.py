import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
import methods
import copy
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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



# def visualize_predictions(data, pred, conf, test_mask, split_idx, true_labels):
#     # Filter predictions for test nodes
#     test_pred = pred[test_mask]
#     test_conf = conf[test_mask]
#     true_labels = data.y[test_mask]
#     test_mask = test_mask.cpu().numpy()
#     # Create confusion matrix
#     cm = confusion_matrix(true_labels.cpu(), test_pred.cpu())

#     # Plotting
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_title(f'Confusion Matrix (Split {split_idx})')
#     ax1.set_xlabel('Predicted Class')
#     ax1.set_ylabel('True Class')

#     # Scatter plot
#     max_conf = torch.max(test_conf, dim=1).values.detach().cpu().numpy()

#     # Determine correctness of predictions
#     correct = test_pred == true_labels

#     # Create a colormap for the two possible cases
#     colormap = {
#         True: 'green',  # Correct prediction
#         False: 'red'    # Wrong prediction
#     }
#     colors = [colormap[c.item()] for c in correct]

#     scatter = ax2.scatter(range(len(max_conf)), max_conf, c=colors, alpha=0.6)
#     ax2.set_title(f'Prediction Confidence (Split {split_idx})')
#     ax2.set_xlabel('Test Sample Index')
#     ax2.set_ylabel('Prediction Confidence')

#     # Count nodes in each category
#     category_counts = {
#         'green': sum(1 for c in colors if c == 'green'),
#         'red': sum(1 for c in colors if c == 'red')
#     }

#     # Create a custom legend with counts
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct: {category_counts["green"]}', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong: {category_counts["red"]}', markerfacecolor='red', markersize=10),
#     ]
#     ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig(f'predictions_split_{split_idx}baseline.png', bbox_inches='tight')
#     plt.close()

#     # Print the counts
#     print(f"Split {split_idx} Node Counts:")
#     print(f"Correct: {category_counts['green']}")
#     print(f"Wrong: {category_counts['red']}")
#     print()

#     return category_counts
    
set_seed(3164711608)


def train_and_get_results(data, model, lr):
    final_test_accuracies = []
    final_val_accuracies = []
    final_train_accuracies = []
    

    def train(model, optimizer):
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

    def val(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())
        return val_acc

    def test(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[test_mask] == data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc, pred, out
 
    for split_idx in range(0, 100):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        model.reset_parameters()
        
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        # Data leakage check
        train_nodes = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(np.intersect1d(train_nodes, test_nodes)) > 0 or len(np.intersect1d(train_nodes, val_nodes)) > 0:
            print(f"Warning: Data leakage detected in split {split_idx}. Skipping this split.")
            continue
        
        print(f"Training for index = {split_idx}")

        train_accuracies = []
        val_accuracies = []
        test_accuracies = []

        for epoch in tqdm(range(1, 101)):
            loss, train_acc = train(model, optimizer)
            #val_acc = val(model)
            test_acc, _, _ = test(model)
            
            train_accuracies.append(train_acc * 100)
            #val_accuracies.append(val_acc * 100)
            test_accuracies.append(test_acc * 100)

        final_test_acc = test_accuracies[-1]
        #final_val_acc = val_accuracies[-1]
        final_train_acc = train_accuracies[-1]

        final_test_accuracies.append(final_test_acc)
        #final_val_accuracies.append(final_val_acc)
        final_train_accuracies.append(final_train_acc)

        print(f"Split {split_idx}: Test Accuracy: {final_test_acc:.2f}%")

        # Visualization (if needed)
        # visualize_predictions(data, pred, out.softmax(dim=1), test_mask, split_idx, data.y.cpu())

    print(f"Average Test Accuracy: {np.mean(final_test_accuracies):.2f}% ± {2 * np.std(final_test_accuracies) / np.sqrt(len(final_test_accuracies)):.2f}%")
    #print(f"Average Validation Accuracy: {np.mean(final_val_accuracies):.2f}% ± {2 * np.std(final_val_accuracies) / np.sqrt(len(final_val_accuracies)):.2f}%")
    print(f"Average Training Accuracy: {np.mean(final_train_accuracies):.2f}% ± {2 * np.std(final_train_accuracies) / np.sqrt(len(final_train_accuracies)):.2f}%")

    return final_test_accuracies, final_val_accuracies, final_train_accuracies

