# SoLAR
Surrogate Label Aware Rewiring for GNNs

Requirements - 

1. Pytorch = 2.2.1
2. Pytorch-Geometric - 2.5.2
3. DGL - 2.1

The folder ``rewiring/`` has spectral rewiring methods to maximize,minimize gap both add/delete.


To run rewiring based on surrogate labels. You can choose different models (GCN, GATv2, SGC).

```Python
python splitrewiring.py --dataset Cora --out 'SolarGCN.csv' --hidden_dimension 32 --LR 0.01 --max_iters_delete 1500
```

To run iterative version of rewiring -

```Python
python main_iter.py --dataset Cora --out 'SolarIterativeGCN.csv' --hidden_dimension 32 --LR 0.01 --max_iters_delete 1500 --train_iters 1
```

