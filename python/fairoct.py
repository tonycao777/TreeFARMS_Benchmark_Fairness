import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/usr/pkg/gurobi/lib/python3.10_utf32/")
from odtlearn.fair_oct import *
from utils import *
import time
import os

def get_fairoct_depth(model, node_dict, node_id=1):
    """
    Returns
    ---
    natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
    """
    _, _, selected_feature, cutoff, leaf, value = node_dict[node_id]
    
    if leaf:
        return 1
    else:
        return 1+max(get_fairoct_depth(model, node_dict, model._tree.get_left_children(node_id)), get_fairoct_depth(model, node_dict, model._tree.get_right_children(node_id)))


def get_fairoct_leaves(model, node_dict, node_id=1):
    leaves_counter = 0
    _, _, selected_feature, cutoff, leaf, value = node_dict[node_id]
    if leaf:
        leaves_counter += 1
    else:
        leaves_counter += get_fairoct_leaves(model, node_dict, model._tree.get_left_children(node_id))
        leaves_counter += get_fairoct_leaves(model, node_dict, model._tree.get_right_children(node_id))
    return leaves_counter


def train_fairoct(dname, metric, lamb, depth, delta):
    X_train, y_train, sensitive_train = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname))
    P = sensitive_train.reshape(-1, 1)
    l = X_train[:,0]

    X_test, y_test, sensitive_test = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-test-binarized.csv".format(dname))
    
    print(metric, lamb, depth, delta, flush=True)

    if metric == "dp":
        fct = FairSPOCT(
            solver="gurobi",
            positive_class=1,
            depth=depth,
            _lambda=lamb,
            time_limit=3600,
            fairness_bound=delta,
            num_threads=None,
            obj_mode="acc",
            verbose=False,
        )
    elif metric == "eopp":
        fct = FairEOppOCT(
            solver="gurobi",
            positive_class=1,
            depth=depth,
            _lambda=lamb,
            time_limit=3600,
            fairness_bound=delta,
            num_threads=None,
            obj_mode="acc",
            verbose=False,
        )
    elif metric == "eodds":
        fct = FairEOddsOCT(
            solver="gurobi",
            positive_class=1,
            depth=depth,
            _lambda=lamb,
            time_limit=3600,
            fairness_bound=delta,
            num_threads=None,
            obj_mode="acc",
            verbose=False,
        )

    s_time = time.time()
    fct.fit(X_train, y_train, P, l)
    train_time = time.time() - s_time

    node_dict = {}
    for node in np.arange(1, fct._tree.total_nodes+1):
        node_dict[node] = fct._get_node_status(fct.b_value, fct.w_value, fct.p_value, node)

    s_time = time.time()
    train_res = get_tree_fairness(fct, X_train, y_train, sensitive_train)
    test_res = get_tree_fairness(fct, X_test, y_test, sensitive_test)
    fairness_cal_time = time.time() - s_time

    res = np.r_[lamb, depth, delta, train_time, fct.optim_gap, fairness_cal_time,
                get_fairoct_depth(fct, node_dict), get_fairoct_leaves(fct, node_dict), 
                train_res, test_res]
 

    outfile = f"results_fair/fairoct_{dname}_{metric}.p"
    if os.path.exists(outfile):
        out = load_file(outfile)
        out[f"{metric}-{lamb}-{depth}-{delta}-fair"] = res
        out[f"{metric}-{lamb}-{depth}-{delta}-tree"] = node_dict
    else:
        out = { 
            f"{metric}-{lamb}-{depth}-{delta}-fair": res,
            f"{metric}-{lamb}-{depth}-{delta}-tree": node_dict
        }
    print(outfile)
    print("Dumped in here!")
    
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.DEFAULT_PROTOCOL)


    








# X = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],
#                 [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
#                 [1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],
#                 [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
# P = np.array([0,0,0,0,1,
#                 0,0,0,1,1,1,1,1,1,
#                 0,0,1,1,1,1,1,
#                 0,0,0,0,0,1,1])
# y = np.array([0,0,0,1,1,
#                 0,1,1,0,1,1,1,1,1,
#                 0,1,0,0,0,1,1,
#                 0,0,0,0,1,0,0])
# P = P.reshape(-1,1)

# l = X[:,0]

# fcl_wo_SP = FairSPOCT(
#     solver="gurobi",
#     positive_class=1,
#     depth=2,
#     _lambda=0.01,
#     time_limit=100,
#     fairness_bound=1,
#     num_threads=None,
#     obj_mode="acc",
#     verbose=False,
# )
# fcl_wo_SP.fit(X, y, P, l)
# a = fcl_wo_SP

# print(
#     pd.DataFrame(
#         fcl_wo_SP.calc_metric(P, fcl_wo_SP.predict(X)).items(),
#         columns=["(p,y)", "P(Y=y|P=p)"],
#     )
# )


