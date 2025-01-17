import numpy as np
import pandas as pd
import pickle
import os
import time
from utils import *
from plot import *
from sklearn.metrics import accuracy_score
from tree_classifier import Tree
import json


def get_rset_fair_results(dname, lamb, depth, eps, guess=False):
    
    
    if guess:
        train = pd.read_csv("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname), sep=" ", header=None)
        X_train, y_train, sensitive_train = train.iloc[:,2:], train.iloc[:,0], train.iloc[:,1].values
        test = pd.read_csv("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname), sep=" ", header=None)
        X_test, y_test, sensitive_test = test.iloc[:,2:], test.iloc[:,0], test.iloc[:,1].values

        filepath = "rset_{}_guess_{}_{}_{}.p".format(dname, lamb, depth, eps)
    else:
        X_train, y_train, sensitive_train = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname))
        X_test, y_test, sensitive_test = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-test-binarized.csv".format(dname))
    
        filepath = "rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
    
    print(filepath, flush=True)
    # load rashomon set 
    res = load_file("/home/users/dc460/TreeFARMSBenchmark/results_rset/{}".format(filepath))
    
    if guess: 
        enc = res["enc"]
        X_train = enc.transform(X_train)
        X_test = enc.transform(X_test)
        X_train, X_test = X_train.values, X_test.values

    model = res["model"]

    ntrees = model.model_set.model_count
    print("ntrees", ntrees, flush=True)

    n_check_trees = min(ntrees, 100000) # if too many trees in Rset, we sample 100,000 trees

    # store tree_id, depth, nleaves, train_acc, test_acc, 
    # train_dp, test_dp, train_tpr, test_tpr, train_fpr, test_fpr, train_equalized_odd, test_equalized_odd
    results = np.zeros((n_check_trees, 13))

    if n_check_trees == ntrees:
        indices = np.arange(ntrees)
    else:
        np.random.seed(0)
        indices = np.sort(np.random.choice(ntrees, n_check_trees, replace=False))

    s_time = time.time()
    for k, i in enumerate(indices):
        tree = Tree(model.model_set.get_tree_at_idx_raw(i)) # get a single tree
        train_res = get_tree_fairness(tree, X_train, y_train, sensitive_train)
        test_res = get_tree_fairness(tree, X_test, y_test, sensitive_test)
        results[k,]=np.r_[i, tree.maximum_depth(), tree.leaves(), train_res, test_res]
    duration = time.time()-s_time
    print("calculation time", duration, flush=True)
        

    outfile = "/home/users/dc460/TreeFARMSBenchmark/results_fair/"+filepath
    out = {
        "rset": "rset_{}_{}_{}_{}".format(dname, lamb, depth, eps),    
        "ntrees": ntrees,
        "n_check_trees": n_check_trees,
        "results": results,
        "time": duration
    }
    
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.DEFAULT_PROTOCOL)

    

def get_dpf_fair_results(dname, depths=[2,3,4,5], deltas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]):
    X_train, y_train, sensitive_train = load_data("dpf/data/{}-train-binarized.csv".format(dname)) 
    X_test, y_test, sensitive_test = load_data("dpf/data/{}-test-binarized.csv".format(dname))

    results = np.array([]).reshape(0, 15)

    for depth in depths:
        for delta in deltas:
            filepath = "dpf/build/trees/{}-{}-{}.json".format(dname, depth, delta)
            # print(filepath, os.path.exists(filepath), flush=True)
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                trees = json.load(f)
            # print(len(trees), flush=True)
            for i, (k, v) in enumerate(trees.items()):
                tree = Tree(v)
                s_time = time.time()
                train_res = get_tree_fairness(tree, X_train, y_train, sensitive_train)
                test_res = get_tree_fairness(tree, X_test, y_test, sensitive_test)
                duration = time.time()-s_time
                res = np.r_[depth, delta, duration, tree.maximum_depth(), tree.leaves(), train_res, test_res]
                # print(res)
                results = np.r_[results, res.reshape(1, 15)]
                # print(results)

    outfile = f"results_fair/dpf_{dname}.p"
    out = { 
        "results": results
    }

    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.DEFAULT_PROTOCOL)



def get_dpf_fair_results_no_delta(dname, depths=[1,2,3,4,5]):
    X_train, y_train, sensitive_train = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname)) 
    X_test, y_test, sensitive_test = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-test-binarized.csv".format(dname))

    results = np.array([]).reshape(0, 15)

    for depth in depths:
        filepath = "/home/users/dc460/TreeFARMSBenchmark/dpf/build/trees/{}-{}.json".format(dname, depth)
        # print(filepath, os.path.exists(filepath), flush=True)
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            trees = json.load(f)
        # print(len(trees), flush=True)
        for i, (k, v) in enumerate(trees.items()):
            tree = Tree(v)
            s_time = time.time()
            train_res = get_tree_fairness(tree, X_train, y_train, sensitive_train)
            test_res = get_tree_fairness(tree, X_test, y_test, sensitive_test)
            duration = time.time()-s_time
            res = np.r_[depth, duration, tree.maximum_depth(), tree.leaves(), train_res, test_res]
            # print(res)
            results = np.r_[results, res.reshape(1, 15)]
            # print(results)

    outfile = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/dpf_no_delta_{dname}.p"
    out = { 
        "results": results
    }

    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.DEFAULT_PROTOCOL)

def get_dpf_results(dname, depths=[2,3,4,5], deltas=[0.01]):
    # Load training and testing data
    X_train, y_train, sensitive_train = load_data(f"dpf/data/{dname}-train-binarized.csv") 
    X_test, y_test, sensitive_test = load_data(f"dpf/data/{dname}-test-binarized.csv")

    # Initialize the dictionary to store results
    results_dict = {}


    # Iterate over depths and deltas
    for depth in depths:
        for delta in deltas:
            filepath = f"dpf/build/trees/{dname}-{depth}-{delta}.json"
            
            # Skip if the file doesn't exist
            if not os.path.exists(filepath):
                print("file does not exist!!!")
                continue

            # Load the decision tree models
            with open(filepath) as f:
                trees = json.load(f)

            # Iterate through each tree in the json file
            for i, (k, v) in enumerate(trees.items()):
                tree = Tree(v)  # Assuming Tree class handles model loading
                s_time = time.time()

                # Get fairness metrics for the test data
                test_res = get_tree_fairness(tree, X_test, y_test, sensitive_test)
                

                # Extract the fairness metrics: dp, eodds, eopp
                accuracy = test_res[0]  # Accuracy
                dp = test_res[1]        # Demographic Parity
                tpr = test_res[2]       # True Positive Rate
                fpr = test_res[3]       # False Positive Rate
                max_odds = test_res[4]  # Maximum Odds

                # Calculate Equal Opportunity Difference (Eopp)
                yhat = tree.predict(X_test)
                eopp = equal_opportunity_difference(y_test, yhat, sensitive_test)

                duration = time.time() - s_time
                # Create a result dictionary for this particular tree
                tree_results = {
                    'accuracy': accuracy,
                    'dp': dp,
                    'eodds': max_odds,
                    'eopp': eopp,
                    'duration': duration,
                    'max_depth': tree.maximum_depth(),
                    'num_leaves': tree.leaves()
                }
                

                # Store the results in the main dictionary under (depth, delta, tree_index) key
                results_dict[depth] = tree_results

    # Output file
    outfile = f"tables/dpf_{dname}.p"
    out = {
        "results": results_dict
    }
    print(results_dict)
    print("number of trees:", len(results_dict))

    # Save the results as a pickle file
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.DEFAULT_PROTOCOL)