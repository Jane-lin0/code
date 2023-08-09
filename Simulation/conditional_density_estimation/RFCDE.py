import numpy as np
import rfcde

# Parameters
n_trees = 1000     # Number of trees in the forest
mtry = 4           # Number of variables to potentially split at in each node
node_size = 20     # Smallest node size
n_basis = 15       # Number of basis functions
bandwidth = 0.2    # Kernel bandwith - used for prediction only
lambda_param = 10  # Poisson Process parameter

# Fit the model
functional_forest = rfcde.RFCDE(n_trees=n_trees, mtry=mtry, node_size=node_size,
                                n_basis=n_basis)
functional_forest.train(x_train, y_train, flamba=lambda_param)

# ... Same as RFCDE for prediction ...