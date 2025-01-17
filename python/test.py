import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from util import *
import os
import seaborn as sns

def load_rashomon_set(dname, lamb, depth, eps, guess=False):
    """
    Loads the Rashomon set from the specified result file.

    Parameters:
    - dname: Dataset name (e.g., 'german-credit')
    - lamb: Regularization parameter
    - depth: Tree depth
    - eps: Rashomon bound multiplier
    - guess: Whether the results are based on a guess (default is False)

    Returns:
    - model: The TREEFARMS model object
    """
    # Construct the file path for the results
    if guess:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{dname}_guess_{lamb}_{depth}_{eps}.p"
    else:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{dname}_{lamb}_{depth}_{eps}.p"
    
    try:
        # Load the model from the pickle file
        with open(rset_filepath, "rb") as f:
            res = pickle.load(f)
        model = res["model"]
        print(f"Model loaded from {rset_filepath}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file {rset_filepath} not found.")
    except Exception as e:
        raise IOError(f"Error loading the results file: {e}")
    
    # Return the loaded model
    return model

def access_trees_from_rashomon_set(model):
    """
    Access and print each tree in the Rashomon set.

    Parameters:
    - model: The loaded TREEFARMS model object
    """
    # Assuming the trees are stored in the 'model_set' attribute of the TREEFARMS model
    print(model.model_set.model_count)
    for idx in range(10):
        tree = model[idx]  # Access tree by index
        print(f"\nTree {idx + 1}:")
        print(tree)  # You can replace this with specific methods to inspect each tree


# Example usage:
dname = "german-credit"  # Dataset name
lamb = 0.01  # Example regularization parameter
depth = 5  # Example tree depth
eps = 0.05  # Example Rashomon bound multiplier

# Load the Rashomon set
model = load_rashomon_set(dname, lamb, depth, eps, guess=False)

# Access the individual trees in the Rashomon set
access_trees_from_rashomon_set(model)