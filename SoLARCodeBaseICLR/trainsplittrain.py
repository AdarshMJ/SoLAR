# import os
# import random
# import time
# import numpy as np
# import torch
# from tqdm import tqdm
# import methods
# import copy
# from spectralclustering import *

#planetoid_val_seeds =  [3164711608]
#planetoid_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]


# def set_seed(seed):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")



# set_seed(3164711608)
# def train_and_get_results(data, ogmodel,ogoptimizer,remodel,reoptimizer,edgesdelete,edgesadd,p):
#     avg_testacc_before = []
#     avg_acc_testallsplits_before = []
#     avg_testacc_after = []
#     avg_acc_testallsplits_after = []


#     criterion = torch.nn.CrossEntropyLoss()

#     def train(model,optimizer):
#         model.train()
#         optimizer.zero_grad()  
#         out = model(data.x, data.edge_index)          
#         loss = criterion(out[train_mask], data.y[train_mask])
#         loss.backward()  
#         optimizer.step()  
#         pred = out.argmax(dim=1)  
#         train_correct = pred[train_mask] == data.y[train_mask]  
#         train_acc = int(train_correct.sum()) / int(train_mask.sum())  
#         return loss


#     def val(model):
#         model.eval()
#         out = model(data.x, data.edge_index)
#         valpred = out.argmax(dim=1)  # Use the class with highest probability. 
#         val_correct = valpred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
#         val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
#         return val_acc,valpred


#     def test(model):
#             model.eval()
#             out= model(data.x, data.edge_index)
#             pred = out.argmax(dim=1)  # Use the class with highest probability. 
#             test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
#             test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
#             return test_acc,pred

#     original_data = copy.deepcopy(data)
#     for split_idx in range(0,100):
#         data = copy.deepcopy(original_data)
#         train_mask = data.train_mask[:,split_idx]
#         test_mask = data.test_mask[:,split_idx]
#         val_mask = data.val_mask[:,split_idx]
        
#         #================ Check for data leakage =================#
#         train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         leakage_nodes = np.intersect1d(train_nodes, test_nodes)
        
#         if len(leakage_nodes) > 0:
#             print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets. Stopping execution.")
#             sys.exit(1)  # Exit the script due to data leakage
        
#         #=========================================================#
        
        
#         print(f"Training for index = {split_idx}")
#         for epoch in tqdm(range(1, 101)):
#             loss = train(ogmodel,ogoptimizer)
#         val_acc,valpred = val(ogmodel)
#         test_acc,pred = test(ogmodel)
#         avg_testacc_before.append(test_acc*100)
#         print(f'Test Accuracy: {test_acc:.2f}')
#         print()
#         avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))
        
#         print()
#         print(f"Rewiring for index = {split_idx} -- Deleting {edgesdelete} and Adding {edgesadd} edges")
#         data, ActualEdgesRemoved, ActualEdgesAdded = methods.PeerGNNDeleteAdd(data, edgesdelete, edgesadd,pred,train_mask)
#         print(data)
#         print()
    
#         print("Start re-training ....")
#         for epoch in tqdm(range(1, 101)):
#            loss = train(remodel,reoptimizer)
#         val_acc,valrepred = val(remodel)
#         test_acc,repred= test(remodel)
#         avg_testacc_after.append(test_acc*100)
#         pred = repred
#         #valpred = valrepred
#         #print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f}')
#         print(f'Test Accuracy: {test_acc:.2f}')
#         print()
#         avg_acc_testallsplits_after.append(np.mean(avg_testacc_after))

#     return avg_acc_testallsplits_after,ActualEdgesRemoved, ActualEdgesAdded


# def visualize_prediction_changes(data, pred_before, pred_after, conf_before, conf_after, test_mask, split_idx,true_labels):
#     # Filter predictions for test nodes
#     test_pred_before = pred_before[test_mask]
#     test_pred_after = pred_after[test_mask]
#     test_conf_before = conf_before[test_mask]
#     test_conf_after = conf_after[test_mask]
#     true_labels = data.y[test_mask]

#     # Create confusion matrix
#     cm = confusion_matrix(test_pred_before.detach().cpu().numpy(), test_pred_after.detach().cpu().numpy())

#     # Plotting
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_title(f'Class Transition Heatmap (Split {split_idx})')
#     ax1.set_xlabel('Predicted Class After Rewiring')
#     ax1.set_ylabel('Predicted Class Before Rewiring')

#     # Scatter plot
#     max_conf_before = torch.max(test_conf_before, dim=1).values
#     max_conf_after = torch.max(test_conf_after, dim=1).values
    
#     # Determine correctness of predictions
#     correct_before = test_pred_before == true_labels
#     correct_after = test_pred_after == true_labels

#     # Create a colormap for the four possible cases
#     colormap = {
#         (True, True): 'green',    # Correct before and after
#         (True, False): 'orange',  # Correct before, wrong after
#         (False, True): 'blue',    # Wrong before, correct after
#         (False, False): 'red'     # Wrong before and after
#     }
    
#     colors = [colormap[(cb.item(), ca.item())] for cb, ca in zip(correct_before, correct_after)]

#     scatter = ax2.scatter(max_conf_before.detach().cpu().numpy(), max_conf_after.detach().cpu().numpy(), c=colors, alpha=0.6)
#     ax2.set_title(f'Confidence Change Scatter Plot (Split {split_idx})')
#     ax2.set_xlabel('Confidence Before Rewiring')
#     ax2.set_ylabel('Confidence After Rewiring')
#     ax2.plot([0, 1], [0, 1], 'k--')  # Diagonal line

#     # Create a custom legend
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label='Correct → Correct', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Correct → Wrong', markerfacecolor='orange', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Wrong → Correct', markerfacecolor='blue', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Wrong → Wrong', markerfacecolor='red', markersize=10),
#     ]
#     ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig(f'prediction_changes_split_{split_idx}.png', bbox_inches='tight')
#     plt.close()



import torch
import numpy as np
import random
import os
import copy
import sys
from tqdm import tqdm
import methods
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import torch.nn.functional as F
from torch_geometric.data import Data



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

# def visualize_prediction_changes(data, pred_before, pred_after, conf_before, conf_after, test_mask, split_idx,true_labels,edgesdelete,edgesadd):
#     # Filter predictions for test nodes
#     test_pred_before = pred_before[test_mask]
#     test_pred_after = pred_after[test_mask]
#     test_conf_before = conf_before[test_mask]
#     test_conf_after = conf_after[test_mask]
#     true_labels = data.y[test_mask]

#     # Create confusion matrix
#     cm = confusion_matrix(test_pred_before.cpu(), test_pred_after.cpu())

#     # Plotting
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_title(f'Class Transition Heatmap (Split {split_idx})')
#     ax1.set_xlabel('Predicted Class After Rewiring')
#     ax1.set_ylabel('Predicted Class Before Rewiring')

#     # Scatter plot
#     max_conf_before = torch.max(test_conf_before, dim=1).values.detach().cpu().numpy()
#     max_conf_after = torch.max(test_conf_after, dim=1).values.detach().cpu().numpy()
    
#     # Determine correctness of predictions
#     correct_before = test_pred_before == true_labels
#     correct_after = test_pred_after == true_labels

#     # Create a colormap for the four possible cases
#     colormap = {
#         (True, True): 'green',    # Correct before and after
#         (True, False): 'orange',  # Correct before, wrong after
#         (False, True): 'blue',    # Wrong before, correct after
#         (False, False): 'red'     # Wrong before and after
#     }
    
#     colors = [colormap[(cb.item(), ca.item())] for cb, ca in zip(correct_before, correct_after)]

#     scatter = ax2.scatter(max_conf_before, max_conf_after, c=colors, alpha=0.6)
#     ax2.set_title(f'Confidence Change Scatter Plot (Split {split_idx})')
#     ax2.set_xlabel('Confidence Before Rewiring')
#     ax2.set_ylabel('Confidence After Rewiring')
#     ax2.plot([0, 1], [0, 1], 'k--')  # Diagonal line

#     # Count nodes in each category
#     category_counts = {
#         'green': sum(1 for c in colors if c == 'green'),
#         'orange': sum(1 for c in colors if c == 'orange'),
#         'blue': sum(1 for c in colors if c == 'blue'),
#         'red': sum(1 for c in colors if c == 'red')
#     }

#     # Create a custom legend with counts
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Correct: {category_counts["green"]}', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Wrong: {category_counts["orange"]}', markerfacecolor='orange', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Correct: {category_counts["blue"]}', markerfacecolor='blue', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Wrong: {category_counts["red"]}', markerfacecolor='red', markersize=10),
#     ]
#     ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig(f'prediction_changes_split_{split_idx}GATv2GCNAdd.png', bbox_inches='tight')
#     plt.close()

#     # Print the counts
#     print(f"Split {split_idx} Node Counts:")
#     print(f"Correct → Correct: {category_counts['green']}")
#     print(f"Correct → Wrong: {category_counts['orange']}")
#     print(f"Wrong → Correct: {category_counts['blue']}")
#     print(f"Wrong → Wrong: {category_counts['red']}")
#     print()

#     return category_counts

# def visualize_prediction_changes(model, data, pred_before, pred_after, conf_before, conf_after, test_mask, split_idx, true_labels, noise_level=0.1):
#     # Filter predictions for test nodes
    
#     test_pred_before = pred_before[test_mask]
#     test_pred_after = pred_after[test_mask]
#     test_conf_before = conf_before[test_mask]
#     test_conf_after = conf_after[test_mask]
#     true_labels = data.y[test_mask]
#     test_mask = test_mask.cpu()
#     # Calculate KD retention
#     delta_entropy = kd_retention(model, data, noise_level)
#     test_delta_entropy = delta_entropy[test_mask]
#     #test_delta_entropy = test_delta_entropy.detach().cpu().numpy()

#     # Determine high delta entropy nodes (e.g., top 10%)
#     high_entropy_threshold = np.percentile(test_delta_entropy, 90)
#     high_entropy_mask = test_delta_entropy >= high_entropy_threshold

#     # Plotting
#     fig, ax = plt.subplots(figsize=(12, 10))

#     # Scatter plot
#     max_conf_before = torch.max(test_conf_before, dim=1).values.detach().cpu().numpy()
#     max_conf_after = torch.max(test_conf_after, dim=1).values.detach().cpu().numpy()

#     # Determine correctness of predictions
#     correct_before = test_pred_before == true_labels
#     correct_after = test_pred_after == true_labels

#     # Create a colormap for the four possible cases
#     colormap = {
#         (True, True): 'green',   # Correct before and after
#         (True, False): 'orange', # Correct before, wrong after
#         (False, True): 'blue',   # Wrong before, correct after
#         (False, False): 'red'    # Wrong before and after
#     }
#     colors = [colormap[(cb.item(), ca.item())] for cb, ca in zip(correct_before, correct_after)]

#     # Plot low entropy nodes
#     low_entropy_scatter = ax.scatter(
#         max_conf_before[~high_entropy_mask],
#         max_conf_after[~high_entropy_mask],
#         c=[c for c, he in zip(colors, high_entropy_mask) if not he],
#         alpha=0.6,
#         marker='o'
#     )

#     # Plot high entropy nodes with a different marker
#     high_entropy_scatter = ax.scatter(
#         max_conf_before[high_entropy_mask],
#         max_conf_after[high_entropy_mask],
#         c=[c for c, he in zip(colors, high_entropy_mask) if he],
#         alpha=0.6,
#         marker='*',
#         s=100  # Larger size for visibility
#     )

#     ax.set_title(f'Confidence Change Scatter Plot (Split {split_idx})')
#     ax.set_xlabel('Confidence Before Rewiring')
#     ax.set_ylabel('Confidence After Rewiring')
#     ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line

#     # Count nodes in each category
#     category_counts = {
#         'green': sum(1 for c in colors if c == 'green'),
#         'orange': sum(1 for c in colors if c == 'orange'),
#         'blue': sum(1 for c in colors if c == 'blue'),
#         'red': sum(1 for c in colors if c == 'red')
#     }

#     # Create a custom legend with counts
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Correct: {category_counts["green"]}', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Wrong: {category_counts["orange"]}', markerfacecolor='orange', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Correct: {category_counts["blue"]}', markerfacecolor='blue', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Wrong: {category_counts["red"]}', markerfacecolor='red', markersize=10),
#         plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
#     ]
#     ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig(f'prediction_changes_split_{split_idx}_with_entropy.png', bbox_inches='tight')
#     plt.close()

#     # Print the counts
#     print(f"Split {split_idx} Node Counts:")
#     print(f"Correct → Correct: {category_counts['green']}")
#     print(f"Correct → Wrong: {category_counts['orange']}")
#     print(f"Wrong → Correct: {category_counts['blue']}")
#     print(f"Wrong → Wrong: {category_counts['red']}")
#     print(f"High Δ Entropy: {sum(high_entropy_mask)}")
#     print()

#     return category_counts

def visualize_prediction_changes(model, data, pred_before, pred_after, conf_before, conf_after, test_mask, split_idx, true_labels, noise_level=0.1):
    # Filter predictions for test nodes
    test_pred_before = pred_before[test_mask]
    test_pred_after = pred_after[test_mask]
    test_conf_before = conf_before[test_mask]
    test_conf_after = conf_after[test_mask]
    true_labels = data.y[test_mask]
    test_mask = test_mask.cpu()
    
    # Calculate KD retention
    delta_entropy = kd_retention(model, data, noise_level)
    test_delta_entropy = delta_entropy[test_mask]

    # Determine high delta entropy nodes (e.g., top 10%)
    high_entropy_threshold = np.percentile(test_delta_entropy, 90)
    high_entropy_mask = test_delta_entropy >= high_entropy_threshold

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    max_conf_before = torch.max(test_conf_before, dim=1).values.detach().cpu().numpy()
    max_conf_after = torch.max(test_conf_after, dim=1).values.detach().cpu().numpy()

    # Determine correctness of predictions
    correct_before = test_pred_before == true_labels
    correct_after = test_pred_after == true_labels

    # Create a colormap for the four possible cases
    colormap = {
        (True, True): 'green',   # Correct before and after
        (True, False): 'orange', # Correct before, wrong after
        (False, True): 'blue',   # Wrong before, correct after
        (False, False): 'red'    # Wrong before and after
    }
    colors = [colormap[(cb.item(), ca.item())] for cb, ca in zip(correct_before, correct_after)]

    # Plot low entropy nodes
    low_entropy_scatter = ax.scatter(
        max_conf_before[~high_entropy_mask],
        max_conf_after[~high_entropy_mask],
        c=[c for c, he in zip(colors, high_entropy_mask) if not he],
        alpha=0.6,
        marker='D'
    )

    # Plot high entropy nodes with a different marker
    high_entropy_scatter = ax.scatter(
        max_conf_before[high_entropy_mask],
        max_conf_after[high_entropy_mask],
        c=[c for c, he in zip(colors, high_entropy_mask) if he],
        alpha=0.6,
        marker='*',
        s=100  # Larger size for visibility
    )

    ax.set_title(f'Confidence Change Scatter Plot (Split {split_idx})')
    ax.set_xlabel('Confidence Before Rewiring')
    ax.set_ylabel('Confidence After Rewiring')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line

    # Count nodes in each category
    category_counts = {
        'green': sum(1 for c in colors if c == 'green'),
        'orange': sum(1 for c in colors if c == 'orange'),
        'blue': sum(1 for c in colors if c == 'blue'),
        'red': sum(1 for c in colors if c == 'red')
    }

    # Count high entropy nodes in each category
    high_entropy_counts = {
        'green': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'green' and he),
        'orange': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'orange' and he),
        'blue': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'blue' and he),
        'red': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'red' and he)
    }

    # Create a custom legend with counts
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Correct: {category_counts["green"]} (High Δ Entropy: {high_entropy_counts["green"]})', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Wrong: {category_counts["orange"]} (High Δ Entropy: {high_entropy_counts["orange"]})', markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Correct: {category_counts["blue"]} (High Δ Entropy: {high_entropy_counts["blue"]})', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Wrong: {category_counts["red"]} (High Δ Entropy: {high_entropy_counts["red"]})', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'prediction_changes_split_{split_idx}_with_entropy.png', bbox_inches='tight')
    plt.close()

    # Print the counts
    print(f"Split {split_idx} Node Counts:")
    print(f"Correct → Correct: {category_counts['green']} (High Δ Entropy: {high_entropy_counts['green']})")
    print(f"Correct → Wrong: {category_counts['orange']} (High Δ Entropy: {high_entropy_counts['orange']})")
    print(f"Wrong → Correct: {category_counts['blue']} (High Δ Entropy: {high_entropy_counts['blue']})")
    print(f"Wrong → Wrong: {category_counts['red']} (High Δ Entropy: {high_entropy_counts['red']})")
    print(f"Total High Δ Entropy: {sum(high_entropy_mask)}")
    print()

    return category_counts, high_entropy_counts




# Assuming the kd_retention function is defined as provided
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



def train_and_get_results(data, ogmodel, oglr, remodel, relr, edgesdelete, edgesadd):
    test_acc_before = []
    test_acc_after = []
    val_acc_after = []
    

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
        return loss

    def val(model):
        model.eval()
        out = model(data.x, data.edge_index)
        valpred = out.argmax(dim=1)
        val_correct = valpred[val_mask] == data.y[val_mask]
        val_acc = int(val_correct.sum()) / int(val_mask.sum())
        return val_acc, valpred, out

    def test(model):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[test_mask] == data.y[test_mask]
        test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc, pred, out

    original_data = copy.deepcopy(data)
    for split_idx in range(0, 100):
        data = copy.deepcopy(original_data)
        criterion = torch.nn.CrossEntropyLoss()
        ogoptimizer = torch.optim.Adam(ogmodel.parameters(), lr=oglr, weight_decay=5e-4)
        reoptimizer = torch.optim.Adam(remodel.parameters(), lr=relr, weight_decay=5e-4)
        ogmodel.reset_parameters()
        #ogoptimizer = type(ogoptimizer)(ogmodel.parameters(), **ogoptimizer.defaults)
        
        remodel.reset_parameters()
        #reoptimizer = type(reoptimizer)(remodel.parameters(), **reoptimizer.defaults)
        
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        # Check for data leakage
        train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        leakage_nodes = np.intersect1d(train_nodes, test_nodes)
        
        if len(leakage_nodes) > 0:
            print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets. Stopping execution.")
            sys.exit(1)
        
        print(f"Training for index = {split_idx}")
        for epoch in tqdm(range(1, 101)):
            loss_b = train(ogmodel, ogoptimizer)
        #val_acc_b, val_pred_b, val_out_b = val(ogmodel)
        test_acc_b, pred_b, test_out_b = test(ogmodel)
        
        test_acc_before.append(test_acc_b * 100)
        #val_acc_before.append(val_acc_b * 100)
        print(f'Test Accuracy: {test_acc_b:.4f}')
        print()
        
        print(f"Rewiring for index = {split_idx} -- Deleting {edgesdelete} and Adding {edgesadd} edges")
        data, ActualEdgesRemoved, ActualEdgesAdded = methods.PeerGNNDeleteAdd(data, edgesdelete, edgesadd, pred_b, train_mask, val_mask)
        print(data)
        print()
    
        print("Start re-training ....")
        for epoch in tqdm(range(1, 101)):
           loss_a = train(remodel, reoptimizer)
        test_acc_a, pred_a, test_out_a = test(remodel)

        test_acc_after.append(test_acc_a * 100)
        val_acc_a, val_pred_a, val_out_a = val(remodel)
        val_acc_after.append(val_acc_a * 100)
        print(f'Test Accuracy: {test_acc_a:.4f}')
        print(f'Validation Accuracy: {val_acc_a:.4f}')
        print()

    visualize_prediction_changes(remodel, data, pred_b, pred_a, test_out_b.softmax(dim=1), test_out_a.softmax(dim=1), test_mask, split_idx, data.y, noise_level=1.0)

    avg_acc_before = np.mean(test_acc_before)
    avg_acc_after = np.mean(test_acc_after)
    avg_val_acc_after = np.mean(val_acc_after)
    
    #print(f"Average Test Accuracy Before: {avg_acc_before:.2f}%")
    print(f"Average Test Accuracy After: {avg_acc_after:.2f}%")
    print(f"Average Validation Accuracy After: {avg_val_acc_after:.2f}%")
    return test_acc_after, val_acc_after, ActualEdgesRemoved, ActualEdgesAdded

# import torch
# import numpy as np
# import random
# import os
# import copy
# import sys
# from tqdm import tqdm
# import methods
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# def visualize_aggregated_prediction_changes(all_data):
#     # Unpack the aggregated data
#     all_pred_before = torch.cat([d['pred_before'] for d in all_data])
#     all_pred_after = torch.cat([d['pred_after'] for d in all_data])
#     all_conf_before = torch.cat([d['conf_before'] for d in all_data])
#     all_conf_after = torch.cat([d['conf_after'] for d in all_data])
#     all_true_labels = torch.cat([d['true_labels'] for d in all_data])

#     # Create confusion matrix
#     cm = confusion_matrix(all_pred_before.cpu(), all_pred_after.cpu())

#     # Plotting
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_title('Aggregated Class Transition Heatmap')
#     ax1.set_xlabel('Predicted Class After Rewiring')
#     ax1.set_ylabel('Predicted Class Before Rewiring')

#     # Scatter plot
#     max_conf_before = torch.max(all_conf_before, dim=1).values.detach().cpu().numpy()
#     max_conf_after = torch.max(all_conf_after, dim=1).values.detach().cpu().numpy()
    
#     # Determine correctness of predictions
#     correct_before = all_pred_before == all_true_labels
#     correct_after = all_pred_after == all_true_labels

#     # Create a colormap for the four possible cases
#     colormap = {
#         (True, True): 'green',    # Correct before and after
#         (True, False): 'orange',  # Correct before, wrong after
#         (False, True): 'blue',    # Wrong before, correct after
#         (False, False): 'red'     # Wrong before and after
#     }
    
#     colors = [colormap[(cb.item(), ca.item())] for cb, ca in zip(correct_before, correct_after)]

#     scatter = ax2.scatter(max_conf_before, max_conf_after, c=colors, alpha=0.6)
#     ax2.set_title('Aggregated Confidence Change Scatter Plot')
#     ax2.set_xlabel('Confidence Before Rewiring')
#     ax2.set_ylabel('Confidence After Rewiring')
#     ax2.plot([0, 1], [0, 1], 'k--')  # Diagonal line

#     # Count nodes in each category
#     category_counts = {
#         'green': sum(1 for c in colors if c == 'green'),
#         'orange': sum(1 for c in colors if c == 'orange'),
#         'blue': sum(1 for c in colors if c == 'blue'),
#         'red': sum(1 for c in colors if c == 'red')
#     }

#     # Create a custom legend with counts
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Correct: {category_counts["green"]}', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct → Wrong: {category_counts["orange"]}', markerfacecolor='orange', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Correct: {category_counts["blue"]}', markerfacecolor='blue', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong → Wrong: {category_counts["red"]}', markerfacecolor='red', markersize=10),
#     ]
#     ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig('aggregated_prediction_changes.png', bbox_inches='tight')
#     plt.close()

#     # Print the counts
#     print("Aggregated Node Counts:")
#     print(f"Correct → Correct: {category_counts['green']}")
#     print(f"Correct → Wrong: {category_counts['orange']}")
#     print(f"Wrong → Correct: {category_counts['blue']}")
#     print(f"Wrong → Wrong: {category_counts['red']}")
#     print()

#     return category_counts

# def train_and_get_results(data, ogmodel, ogoptimizer, remodel, reoptimizer, edgesdelete, edgesadd):
#     avg_testacc_before = []
#     avg_acc_testallsplits_before = []
#     avg_testacc_after = []
#     avg_acc_testallsplits_after = []
#     all_visualization_data = []

#     criterion = torch.nn.CrossEntropyLoss()

#     def train(model, optimizer):
#         model.train()
#         optimizer.zero_grad()  
#         out = model(data.x, data.edge_index)          
#         loss = criterion(out[train_mask], data.y[train_mask])
#         loss.backward()  
#         optimizer.step()  
#         pred = out.argmax(dim=1)  
#         train_correct = pred[train_mask] == data.y[train_mask]  
#         train_acc = int(train_correct.sum()) / int(train_mask.sum())  
#         return loss

#     def val(model):
#         model.eval()
#         out = model(data.x, data.edge_index)
#         valpred = out.argmax(dim=1)
#         val_correct = valpred[val_mask] == data.y[val_mask]
#         val_acc = int(val_correct.sum()) / int(val_mask.sum())
#         return val_acc, valpred, out

#     def test(model):
#         model.eval()
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         test_correct = pred[test_mask] == data.y[test_mask]
#         test_acc = int(test_correct.sum()) / int(test_mask.sum())
#         return test_acc, pred, out

#     original_data = copy.deepcopy(data)
#     for split_idx in range(0, 100):
#         data = copy.deepcopy(original_data)
#         train_mask = data.train_mask[:, split_idx]
#         test_mask = data.test_mask[:, split_idx]
#         val_mask = data.val_mask[:, split_idx]
        
#         # Check for data leakage
#         train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
#         leakage_nodes = np.intersect1d(train_nodes, test_nodes)
        
#         if len(leakage_nodes) > 0:
#             print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets. Stopping execution.")
#             sys.exit(1)
        
#         print(f"Training for index = {split_idx}")
#         for epoch in tqdm(range(1, 101)):
#             loss_b = train(ogmodel, ogoptimizer)
#         val_acc_b, valpred_b, val_out_b = val(ogmodel)
#         test_acc_b, pred_b, test_out_b = test(ogmodel)
#         avg_testacc_before.append(test_acc_b*100)
#         print(f'Test Accuracy: {test_acc_b:.4f}')
#         print()
#         avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))
        
#         print()
#         print(f"Rewiring for index = {split_idx} -- Deleting {edgesdelete} and Adding {edgesadd} edges")
#         data, ActualEdgesRemoved, ActualEdgesAdded = methods.PeerGNNDeleteAdd(data, edgesdelete, edgesadd, pred_b, train_mask)
#         print(data)
#         print()
    
#         print("Start re-training ....")
#         for epoch in tqdm(range(1, 101)):
#            loss_a = train(remodel, reoptimizer)
#         val_acc_a, valpred_a, val_out_a = val(remodel)
#         test_acc_a, pred_a, test_out_a = test(remodel)
#         avg_testacc_after.append(test_acc_a*100)
#         print(f'Test Accuracy: {test_acc_a:.4f}')
#         print()
#         avg_acc_testallsplits_after.append(np.mean(avg_testacc_after))
#         visualization_data = {
#             'pred_before': pred_b[test_mask],
#             'pred_after': pred_a[test_mask],
#             'conf_before': test_out_b.softmax(dim=1)[test_mask],
#             'conf_after': test_out_a.softmax(dim=1)[test_mask],
#             'true_labels': data.y[test_mask]
#         }
#         all_visualization_data.append(visualization_data)
    
#     visualize_aggregated_prediction_changes(all_visualization_data)
#     return avg_acc_testallsplits_after, ActualEdgesRemoved, ActualEdgesAdded


