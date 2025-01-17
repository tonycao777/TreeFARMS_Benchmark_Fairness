import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from threshold_guess import *
from gosdt import GOSDTClassifier

import sys

# Redirect stdout to a file
sys.stdout = open('/home/users/dc460/TreeFARMSBenchmark/python/output.txt', 'w')


# Step 1: Split dataset into training set, selection set, and test set

# Function to  split into training, selection, and test set
def set_up(dname):
    # Load the dataset using the load_data function
    X, y, sensitive_train = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname))
    
    # Split into training set (80%) and selection set (20%)
    X_train, X_select, y_train, y_select, sensitive_train_split, sensitive_select_split = train_test_split(X, y, sensitive_train, test_size=0.2, random_state=42)
    
    return X_train, X_select, y_train, y_select, sensitive_train_split, sensitive_select_split

# Step 2: Do GridSearch on GOSDT parameters to determine the parameters that lead to best accuracy on the training set. (We get a specific lambda and a depth_limit)

# Function to do GridSearch on the Training set
def cross_validate_gosdt(X, y, depth, lamb, cv = 5):

    # Prepare the hyperparameter grid
    param_grid = {
        'depth_budget': depth,
        'regularization': lamb
    }

    # Initialize the GOSDT classifier
    model = GOSDTClassifier()
    print("sets up the model")
    # Should just do threshold guessing for depth 5
    # Can preprocess GC
    # Might be better to just manually implement 
    # Set up GridSearch
    grid_search = GridSearchCV(estimator = model, param_grid=param_grid, cv = cv, scoring = 'accuracy', verbose = 3)
    
    # Perform the grid search
    grid_search.fit(X, y)
    print("performed grid search")

    # Get the best parameters and scores
    best_param = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_param}")
    print(f"Best Cross-Validation Score: {best_score}")

    return best_param, best_score, grid_search

# Step 3: Obtain the Rset using the lambda and depth_limit we got from step 2. We can obtain a few sets based on different Epsilon value.
# Use get_rset from utils

# Step 4: Evaluate every tree in each Rset using the selection set based on fairness metric, and in the end pick the tree that offers the best fairness metric (dp, eodd, eopp).


# Function to select the best tree from the rset

def select_best(dname, regularization, depth, epsilon, X_select, y_select, sensitive_select_split, guess = False, f_metric = [demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference]):

    # Load the Rset
    depth = depth + 1
    if guess == True:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{dname}_guess_{regularization}_{depth}_{epsilon}.p"
    else:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{dname}_{regularization}_{depth}_{epsilon}.p"
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
    
    ntrees = model.model_set.model_count
    n_check_trees = min(ntrees, 100000) # if too many trees in Rset, we sample 100,000 trees
    print("There are ", n_check_trees, "trees to select from!")

    if isinstance(X_select, np.ndarray):  # If it's a numpy array
        X_select = pd.DataFrame(X_select)  # Convert to DataFrame

    
    best_trees = {demographic_parity_difference: [None, 0], equal_opportunity_difference: [None, 0], equalized_odds_difference: [None, 0]}
    for metric in f_metric:
        # Initialize variables to track the best model and its performance
        best_fmetric_value = float('inf')  # Assuming you want to minimize the fairness metric
        best_tree = None

        for idx in range(n_check_trees):
            tree = model[idx]  # Indexing to get each tree in the Rashomon set
            
            # Evaluate the tree on the selection set using the fairness metric
            yhat = tree.predict(X_select)
            if (metric == equalized_odds_difference):
                _, _, fairness_value = metric(y_select, yhat, sensitive_select_split)
            else:
                fairness_value = metric(y_select, yhat, sensitive_select_split)
            acc = accuracy_score(y_select, yhat)
      
            # Track the best tree based on the fairness metric
            if (1-acc) + 0.3*fairness_value < best_fmetric_value:  # If you're minimizing the fairness metric
                best_fmetric_value = fairness_value
                best_tree = tree
        
        best_trees[metric][0] = best_tree
        best_trees[metric][1] = best_fmetric_value

    return best_trees

# Function to evaluate the tree on the test set
def evaluate(dname, best_trees):
    X_test, y_test, sensitive_test = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-test-binarized.csv".format(dname))
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store accuracy and fairness scores for each tree
    all_metrics = {metric: {'accuracies': [], 'fairness': []} for metric in best_trees}

    if isinstance(X_test, np.ndarray):  # If it's a numpy array
        X_test = pd.DataFrame(X_test)  # Convert to DataFrame
    if isinstance(y_test, np.ndarray):  # If it's a numpy array
        y_test = pd.DataFrame(y_test)
    if isinstance(sensitive_test, np.ndarray):  # If it's a numpy array
        sensitive_test = pd.DataFrame(sensitive_test)
    


    # Loop over each fold
    for train_index, test_index in kf.split(X_test):
        # Split data into train and test folds, we don't actually use the train
        X_train, X_test_fold = X_test.iloc[train_index], X_test.iloc[test_index]
        y_train, y_test_fold = y_test.iloc[train_index], y_test.iloc[test_index]
        sensitive_train, sensitive_test_fold = sensitive_test.iloc[train_index], sensitive_test.iloc[test_index]

        # Evaluate each tree on the current fold
        for metric, model in best_trees.items():
            # Predict on the test fold using the current tree
            y_hat = model[0].predict(X_test_fold)
            
            # Compute accuracy for the current fold
            accuracy = accuracy_score(y_test_fold, y_hat)
            all_metrics[metric]['accuracies'].append(accuracy)
            
            # Compute fairness using the provided fairness metric
            if metric == 'equalized_odds_difference':  # Example fairness metric
                # Replace with actual fairness metric
                _, _, fairness_value = metric(y_test_fold, y_hat, sensitive_test_fold)
            else:
                fairness_value = metric(y_test_fold, y_hat, sensitive_test_fold)
            
            # Store the fairness score for the current fold
            all_metrics[metric]['fairness'].append(fairness_value)

    # Calculate mean and std for each tree
    mean_accuracy = {}
    std_accuracy = {}
    mean_fairness = {}
    std_fairness = {}

    for metric in best_trees:
        mean_accuracy[metric] = np.mean(all_metrics[metric]['accuracies'])
        std_accuracy[metric] = np.std(all_metrics[metric]['accuracies'])
        mean_fairness[metric] = np.mean(all_metrics[metric]['fairness'])
        std_fairness[metric] = np.std(all_metrics[metric]['fairness'])

    return mean_accuracy, std_accuracy, mean_fairness, std_fairness


def cross_validation(dname, delta, depth = [2, 3, 4, 5], l = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]):
    # Step 1: Split dataset into training set, selection set, and test set
    X_train, X_select, y_train, y_select, sensitive_train_split, sensitive_select_split = set_up(dname)

    print("Completed setup")
    
    # If depth is >=4, apply threshold guessing       
    X_train_transformed, X_select_transformed = X_train, X_select
    if max(depth) >= 4:
        thresholder = threshold_guess(max_depth=max(depth), n_estimators=100, learning_rate=0.1, backselect=True)
        thresholder.fit(X_train, y_train)
        
        # Transform training and selection datasets based on the learned thresholds
        X_train_transformed = thresholder.transform(X_train)
        X_select_transformed = thresholder.transform(X_select)
        print("Threshold Guessing applied for depth >= 4")

    # Step 2: Do GridSearch on GOSDT parameters to determine the parameters that lead to best accuracy on the training set. (We get a specific lambda and a depth_limit)
    best_param, best_score, grid_search = cross_validate_gosdt(X_train_transformed, y_train, depth, l)

    print("Here are the best Params:", best_param)

    # Step 3: Obtain the Rset using the lambda and depth_limit we got from step 2. We can obtain a few sets based on different Epsilon value.
    get_rset(dname, best_param["regularization"], best_param["depth_budget"], delta, guess = False, max_depth=2, n_est=50, lr=0.1, backselect=True, random_seed=42)

    print("Got the Rset")

    # Step 4: Evaluate every tree in each Rset using the selection set based on fairness metric, and in the end pick the tree that offers the best fairness metric (dp, eodd, eopp).
    best_trees = select_best(dname, best_param["regularization"], best_param["depth_budget"], delta, X_select_transformed, y_select, sensitive_select_split, f_metric = {demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference})

    print("Here's the best Tree:", best_trees)

    mean_accuracy, std_accuracy, mean_fairness, std_fairness = evaluate(dname, best_trees)
    print("the fair score has mean:", mean_fairness, "with standard deviation:", std_fairness)
    print("the accuracy is:", mean_accuracy, "with std:", std_accuracy)
