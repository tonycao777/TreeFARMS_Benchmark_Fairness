import numpy as np
import pandas as pd
import time
from treefarms import TREEFARMS
# from treefarms.model.model_set import ModelSetContainer
# from treefarms.model.tree_classifier import TreeClassifier # Import the tree classification model
import pickle
from sklearn.metrics import accuracy_score
from threshold_guess import *


def get_rset(dname, lamb, depth, eps, guess = False, max_depth=2, n_est=50, lr=0.1, backselect=True, random_seed=42):
    
    df = pd.read_csv("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname), sep=" ")
    X, y = df.iloc[:,2:], df.iloc[:,0]
    print(X.shape, y.shape)

    if guess:
        enc = threshold_guess(max_depth=max_depth, n_estimators=n_est, 
                              learning_rate=lr, backselect=backselect, random_seed=random_seed)
        enc.fit(X, y)
        X = enc.transform(X)
    

    config = {
        "regularization": lamb,  # regularization penalizes the tree with more leaves. We recommend to set it to relative high value to find a sparse tree.
        "depth_budget": depth,
        "rashomon_bound_multiplier": eps,  # rashomon bound multiplier indicates how large of a Rashomon set would you like to get
        "time_limit": 3600,
    }

    model = TREEFARMS(config)
    s_time = time.time()
    model.fit(X, y)
    duration = time.time()-s_time
    ntrees = model.model_set.model_count

    if guess:
        outfile = "/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{}_guess_{}_{}_{}.p".format(dname, lamb, depth, eps)
    else:
        outfile = "/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
    print(outfile)
    print("duration", duration)
    print(ntrees, flush=True)

    for i in model.model_set.available_metrics["metric_values"]:
        print("acc", 1-i[1]/X.shape[0])

    #first_tree = model[0]

    #print("evaluating the first model in the Rashomon set", flush=True)

    # get the results
    #train_acc = first_tree.score(X, y)
    #n_leaves = first_tree.leaves()

    #print("Training accuracy: {}".format(train_acc))
    #print("# of leaves: {}".format(n_leaves))
    #print(first_tree)

    if guess:
        res = {
            "enc": enc, 
            "max_depth": max_depth,
            "n_est": n_est,
            "lr": lr,
            "backselect": backselect,
            "random_seed": random_seed,
            "time": duration,
            "model": model
        }
    
    else:
        res = {
            "time": duration, 
            "model": model
        }

    with open(outfile, "wb") as out:
        pickle.dump(res, out, protocol=pickle.DEFAULT_PROTOCOL)



def load_data(datapath):
    df = pd.read_csv(datapath, sep=" ", header=None)
    X, y = df.values[:,2:], df.values[:,0]
    sensitive_feature = df.values[:,1]
    return X, y, sensitive_feature


def load_file(filepath):
    with open(filepath, "rb") as f:
        out = pickle.load(f)
    return out


# fairness metrics for binary classifier
def demographic_parity_difference(y, yhat, sensitive_features):
    '''
    y, yhat, sensitive_features are all length n array. 
    '''
    uniq = np.unique(sensitive_features)
    probs = np.zeros(len(uniq))
    for i, u in enumerate(uniq):
        idx = np.where(sensitive_features == u)[0]
        probs[i] = yhat[idx].sum()/len(idx)
    diff = np.abs(np.max(probs) - np.min(probs))
    return diff

def fair_tpr_fpr(y, yhat, sensitive_features):
    '''
    y, yhat, sensitive_features are all length n array. 
    '''
    uniq = np.unique(sensitive_features)
    tprs = np.zeros(len(uniq))
    fprs = np.zeros(len(uniq))
    for i, u in enumerate(uniq):
        idx_tp = np.where((sensitive_features == u) & (y == 1))[0]
        tprs[i] = yhat[idx_tp].sum()/len(idx_tp)

        idx_fp = np.where((sensitive_features == u) & (y == 0))[0]
        fprs[i] = yhat[idx_fp].sum()/len(idx_fp)
    return tprs, fprs

def equal_opportunity_difference(y, yhat, sensitive_features):
    tpr, _ = fair_tpr_fpr(y, yhat, sensitive_features)
    diff = np.abs(np.max(tpr) - np.min(tpr))
    return diff

def equalized_odds_difference(y, yhat, sensitive_features):
    tpr, fpr = fair_tpr_fpr(y, yhat, sensitive_features)
    diff_tpr = np.abs(np.max(tpr) - np.min(tpr))
    diff_fpr = np.abs(np.max(fpr) - np.min(fpr))
    return diff_tpr, diff_fpr, max(diff_tpr, diff_fpr)

def get_tree_fairness(tree, X, y, sf):
    '''
    tree: a decision tree in Rset in Tree structure
    X, y: (n,p) matrix like array, (n,) matrix like array
    sf: sensitive feature, (n,) matrix like array
    '''
    yhat = tree.predict(X)
    acc = accuracy_score(y, yhat)
    dp = demographic_parity_difference(y, yhat, sf)
    tpr, fpr, max_odds = equalized_odds_difference(y, yhat, sf)
    return np.array([acc, dp, tpr, fpr, max_odds])