import numpy as np
import pandas as pd
import pickle
import time
from utils import *
from plot import *
from fairoct import *
from cross_val import *
from post import *
from sklearn.metrics import accuracy_score
from tree_classifier import Tree
import json
import sklearn.model_selection 
import importlib.util
import sys
import os

# Add the directory containing get_fair_tree.py to the Python path
module_dir = '/home/users/dc460/TreeFARMSBenchmark/dpf/build'  # Directory containing the module
sys.path.append(module_dir)

# Now you can import the functions from get_fair_tree.py
from get_fair_tree import *


def get_results(depth):
    final_result = {}
    #dname = ["adult", "bank", "compas-recid"] 
    #dname = ["german-credit", "oulad", "student-mat", "student-por"]
    #dname = ["student-mat", "student-por"]
    dname = ["census-income","communities"]
    for name in dname:
        result = get_result(depth, name)
        results = summarize_results(result)
        final_result[name] = results
    
    return final_result
    
def get_result(depth, dname):
    starttime = time.time()
    """
    Args:
    - depth: The depth of the tree to evaluate.
    - dname: The dataset name or identifier.
    - num_trials: Number of trials to repeat the process. Default is 5.
    
    Returns:
    - A dictionary containing results for DPF, FairOCT, RSET, and Post-process methods.
    """
    results = {
        "dpf": {"accuracy":[], "dp":[]},
        "fairoct": {"accuracy":[], "dp":[], "eodds":[], "eopp":[]},
        "post": {"accuracy":[], "dp":[], "eodds":[], "eopp":[]},
        "rset": {"accuracy":[], "dp":[], "eodds":[], "eopp":[]}
    }
    metrics = ["dp", "eodds", "eopp"]
    random_seed = [42, 24, 100, 200, 300]

    for seed in random_seed:
        # Step 1: Split the data into Train, Select, and Test Sets
        X, y, sensitive = load_data(f"/home/users/dc460/TreeFARMSBenchmark/dpf/data/{dname}-binarized.csv")
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = sklearn.model_selection.train_test_split(X, y, sensitive, test_size=0.2, random_state=seed)

        X_train, X_select, y_train, y_select, sensitive_train, sensitive_select = sklearn.model_selection.train_test_split(X_train, y_train, sensitive_train, test_size=0.2, random_state=seed)


        # Step 2: Store my training set so that it can be accessed by run_dpf
        training_set = pd.concat([pd.DataFrame(y_train),  pd.DataFrame(sensitive_train), pd.DataFrame(X_train)], axis=1)
        print("training set shape:", training_set.shape)
        training_set_path = f"/home/users/dc460/TreeFARMSBenchmark/dpf/data/{dname}-train-binarized.csv"
        training_set.to_csv(training_set_path, sep=" ", index=False, header=False)
        print("training data saved")

        testing_set = pd.concat([pd.DataFrame(y_test),  pd.DataFrame(sensitive_test), pd.DataFrame(X_test)], axis=1)
        testing_set_path = f"/home/users/dc460/TreeFARMSBenchmark/dpf/data/{dname}-test-binarized.csv"
        testing_set.to_csv(testing_set_path, sep=" ", index=False, header=False)
        print("testing data saved")
    
        # Step 2: Fit and evaluate DPF
        s_time = time.time()
        run_dpf(dname, depth, 0.01)
        training_time = time.time()-s_time
        print("DPF training time:", training_time)
        dpf_acc, dpf_dp = evaluate_DPF(dname, depth, X_train, y_train, X_test, y_test, sensitive_train, sensitive_test, 0.01)
        results["dpf"]["accuracy"].append(dpf_acc)
        results["dpf"]["dp"].append(dpf_dp)
        

        # Step 3: If at depth 2, then we do FairOCT
        if depth == 2:
            fairoct_accuracy = []
            for metric in metrics:
                fct = fit_fairoct(dname, metric, 0.01, depth, 0.01)
                yhat = fct.predict(X_test)
                if metric == "dp":
                    acc = accuracy_score(y_test, yhat)
                    fairoct_accuracy.append(acc)
                    dp = demographic_parity_difference(y_test, yhat, sensitive_test)
                    results["fairoct"]["dp"].append(dp)
                elif metric == "eodds":
                    acc = accuracy_score(y_test, yhat)
                    fairoct_accuracy.append(acc)
                    tpr, fpr, max_odds = equalized_odds_difference(y_test, yhat, sensitive_test)
                    results["fairoct"]["eodds"].append(max_odds)
                else:
                    acc = accuracy_score(y_test, yhat)
                    fairoct_accuracy.append(acc)
                    eopp = equal_opportunity_difference(y_test, yhat, sensitive_test)
                    results["fairoct"]["eopp"].append(max_odds)
            results["fairoct"]["accuracy"].append(np.mean(fairoct_accuracy))

        # Step 4: Fit RSET and find best trees    
        rset_time = time.time()
        get_rset(dname, 0.01, depth+1, 0.05, guess = False, random_seed=seed)
        f_metric = [demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference]
        # Store rset and f_metric in separate file, pickle dump treeFARM instance. Save at user.xtmp. Store the treeFARM instance
        best_trees = select_best(dname, 0.01, depth, 0.05, X_select, y_select, sensitive_select, False, f_metric)
        rset_accuracy = []
        for metric in f_metric:
            acc, fairness = evaluate_rset(dname, best_trees[metric][0], metric)
            rset_accuracy.append(acc)
            if metric == demographic_parity_difference:
                results["rset"]["dp"].append(fairness)
            elif metric == equalized_odds_difference:
                results["rset"]["eodds"].append(fairness)
            else:
                results["rset"]["eopp"].append(fairness)
        rset_training_time = time.time()-rset_time
        print("RSET training time:", rset_training_time)

        results["rset"]["accuracy"].append(np.mean(rset_accuracy))




        # Attribute-Blind Post-Processing
        post_accuracy = []
        for metric in metrics:
            post_result = post_process(metric, X_train, X_select, X_test, y_train, y_select, y_test, sensitive_train, sensitive_select, sensitive_test)
            #post_result looks like [error rate, fairness, post_processor]
            post_results = evaluate_post(post_result, metric, X_test, y_test, sensitive_test)
            post_accuracy.append(post_results[0])
            results["post"][metric].append(post_results[1])
        
        results["post"]["accuracy"].append(np.mean(post_accuracy))

    print("total result generation time:", time.time() - starttime)
    print(results)

    return results
        


# Helper functions
def evaluate_DPF(dname, depth, X_train, y_train, X_test, y_test, sensitive_train, sensitive_test, delta = 0.01):
    filepath = "dpf/build/trees/{}-{}-{}.json".format(dname, depth, delta)
    if not os.path.exists(filepath):
        print("no path found for the DPF Tree")
        return
    with open(filepath) as f:
        trees = json.load(f)
    print(len(trees))
    acc = 0
    dp = 1
    for i, (k, v) in enumerate(trees.items()):
        tree = Tree(v)
        yhat = tree.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        dp = demographic_parity_difference(y_test, yhat, sensitive_test)
    
    return acc, dp

def fit_fairoct(dname, metric, lamb, depth, delta):
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

    fct.fit(X_train, y_train, P, l)

    fairoct_result = get_tree_fairness(fct, X_test, y_test, sensitive_test)
    return fct

def evaluate_rset(dname, tree, metric):
    X_test, y_test, sensitive_test = load_data("/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-test-binarized.csv".format(dname))
    yhat = tree.predict(pd.DataFrame(X_test))
    acc = accuracy_score(y_test, yhat)
    fairness = metric(y_test, yhat, sensitive_test)
    return acc, fairness

def post_process(metric, X_train, X_select, X_test, y_train, y_select, y_test, sensitive_train, sensitive_select, sensitive_test):
    n_classes = len(np.unique(y_train))
    n_groups = len(np.unique(sensitive_train))
    
    predictor_y = GradientBoostingClassifier()
    predictor_y.fit(X_train, y_train)

    predictor_a = GradientBoostingClassifier()
    predictor_a.fit(X_train, sensitive_train)

    predictor_ay = GradientBoostingClassifier()
    predictor_ay.fit(X_train, sensitive_train*3 + y_train)

    # Define predict functions
    predict_y = lambda x: predictor_y.predict_proba(x)
    predict_a = lambda x: predictor_a.predict_proba(x)
    predict_ay = lambda x: predictor_ay.predict_proba(x)

    if metric == "dp":
        alphas = [0.001, 0.01, 0.1, 0.5]
        # Depending on my accuracy -> go to as high as 0.5
        post_results = []
        for alpha in alphas:
            postprocessor_dp = postprocess.PostProcessor(
            n_classes,
            n_groups,
            pred_y_fn=predict_y,
            pred_a_fn=predict_a,
            criterion='sp',
            alpha=alpha,
            )
            
            postprocessor_dp.fit(X_select, solver=None)

            # Evaluate
            preds_fair = postprocessor_dp.predict(X_select)

            y_select = y_select.astype(int)
            sensitive_select = sensitive_select.astype(int)

            res = []
            res.append(util.error_rate(y_select, preds_fair))
            res.append(util.delta_sp(preds_fair, sensitive_select, n_classes, n_groups))
            res.append(postprocessor_dp)
            post_results.append(res)

        best_result = float('inf')
        best_processor = post_results[0][2]
        for post_result in post_results:
            if post_result[0] + post_result[1]*0.3 < best_result:
                best_result = post_result[0] + post_result[1]*0.3
                best_processor = post_result[2]

        return best_processor

    elif metric == "eodds":
        alphas = [0.001, 0.01, 0.1, 0.5]
        # Depending on my accuracy -> go to as high as 0.5
        post_results = []
        for alpha in alphas:
            postprocessor_eo = postprocess.PostProcessor(
            n_classes,
            n_groups,
            pred_ay_fn=predict_ay,
            criterion='eo',
            alpha=alpha,
            )
            
            postprocessor_eo.fit(X_select, solver=None)

            # Evaluate
            preds_fair = postprocessor_eo.predict(X_select)

            y_select = y_select.astype(int)
            sensitive_select = sensitive_select.astype(int)

            res = []
            res.append(util.error_rate(y_select, preds_fair))
            res.append(util.delta_eo(y_select, preds_fair, sensitive_select, n_classes, n_groups))
            res.append(postprocessor_eo)
            post_results.append(res)

        best_result = float('inf')
        best_processor = post_results[0][2]
        for post_result in post_results:
            if post_result[0] + post_result[1]*0.3 < best_result:
                best_result = post_result[0] + post_result[1]*0.3
                best_processor = post_result[2]

        return best_processor

    else:
        alphas = [0.001, 0.01, 0.1, 0.5]
        # Depending on my accuracy -> go to as high as 0.5
        post_results = []
        for alpha in alphas:
            postprocessor_eopp = postprocess.PostProcessor(
            n_classes,
            n_groups,
            pred_ay_fn=predict_ay,
            criterion='eopp',
            alpha=alpha,
            )
            
            postprocessor_eopp.fit(X_select, solver=None)

            # Evaluate
            preds_fair = postprocessor_eopp.predict(X_select)

            y_select = y_select.astype(int)
            sensitive_select = sensitive_select.astype(int)

            res = []
            res.append(util.error_rate(y_select, preds_fair))
            res.append(util.delta_eopp(y_select, preds_fair, sensitive_select, n_classes, n_groups))
            res.append(postprocessor_eopp)
            post_results.append(res)

        best_result = float('inf')
        best_processor = post_results[0][2]
        for post_result in post_results:
            if post_result[0] + post_result[1]*0.3 < best_result:
                best_result = post_result[0] + post_result[1]*0.3
                best_processor = post_result[2]

        return best_processor

def evaluate_post(results, metric, X_test, y_test, sensitive_test):

    preds_fair = results.predict(X_test)
    post_results = []
    if metric == "dp":
        post_results.append(1-util.error_rate(y_test, preds_fair))
        post_results.append(util.delta_sp(preds_fair, sensitive_test, 2, 2))
    elif metric == "eodds":
        post_results.append(1-util.error_rate(y_test, preds_fair))
        post_results.append(util.delta_eo(y_test, preds_fair, sensitive_test, 2, 2))
    else:
        post_results.append(1-util.error_rate(y_test, preds_fair))
        post_results.append(util.delta_eopp(y_test, preds_fair, sensitive_test, 2, 2))
    
    return post_results

def summarize_results(results):

    for method, metrics in results.items():
        for metric, values in metrics.items():
            if values:  # Check if the list is not empty
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                # Replace the original list with a dictionary of mean and std
                metrics[metric] = {"mean": mean_value, "std": std_value}

    print(results)
    return results

