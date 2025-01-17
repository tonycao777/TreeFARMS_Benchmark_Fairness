import numpy as np
import sklearn, sklearn.linear_model
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import pickle
import os
import time
from utils import *
from plot import *
from sklearn.metrics import accuracy_score
from tree_classifier import Tree
from sklearn.tree import DecisionTreeClassifier
import json
import sys
import os
from gosdt import GOSDTClassifier
# Add the parent directory (TreeFARMSBenchmark) to the Python path
print("Current Working Directory:", os.getcwd())

sys.path.append('/home/users/dc460/TreeFARMSBenchmark/fair-classification')

import postprocess 
import util

def post(dname, depth, l, metric):

    X, y, sensitive = load_data(f"/home/users/dc460/TreeFARMSBenchmark/dpf/data/{dname}-binarized.csv")
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = sklearn.model_selection.train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42  # You can set a fixed random state for reproducibility
    )

    # Second split: 50% of the train data for pre-training and post-training
    inputs_train, inputs_test, labels_train, labels_test, groups_train, groups_test = sklearn.model_selection.train_test_split(
        X_train, y_train, sensitive_train, test_size=0.5, random_state=42
    )

    n_classes = len(np.unique(labels_train))
    n_groups = len(np.unique(groups_train))

#Use GOSDT trees
# Can also use Gradient Boosting - Need to finetune
    predictor_y = GradientBoostingClassifier()
    predictor_y.fit(inputs_train, labels_train)

    predictor_a = GradientBoostingClassifier()
    predictor_a.fit(inputs_train, groups_train)

    predictor_ay = GradientBoostingClassifier()
    predictor_ay.fit(inputs_train, groups_train*3 + labels_train)

    # Define predict functions
    predict_y = lambda x: predictor_y.predict_proba(x)
    predict_a = lambda x: predictor_a.predict_proba(x)
    predict_ay = lambda x: predictor_ay.predict_proba(x)

    # Initialize results dictionary for storing results
    results_fair = {
        'original_accuracy': [],
        'postprocessed_accuracy': [],
        'original_metric': [],
        'postprocessed_metric': []
    }

    ## Post-process for binary equal opportunity
    if metric == "dp":
        alphas = [0.001, 0.01, 0.1, 0.5, 1000]
        # Depending on my accuracy -> go to as high as 0.5
        for alpha in alphas:
            postprocessor_dp = postprocess.PostProcessor(
            n_classes,
            n_groups,
            pred_y_fn=predict_y,
            pred_a_fn=predict_a,
            criterion='sp',
            alpha=alpha,
            )
            
            postprocessor_dp.fit(inputs_test, solver=None)

            # Evaluate
            preds = np.argmax(predict_y(X_test), axis=1)
            preds_fair = postprocessor_dp.predict(X_test)

            # Ensure y_test, preds, and sensitive_test are all integers
            y_test = y_test.astype(int)
            preds = preds.astype(int)
            sensitive_test = sensitive_test.astype(int)


            # Debugging: Check the shapes and types of y_preds and groups
            print("y_preds type:", type(y_test), "y_preds shape:", y_test.shape)
            print("groups type:", type(preds), "groups shape:", preds.shape)
    

            # Save results
            results_fair['original_accuracy'].append(1 - util.error_rate(y_test, preds))
            results_fair['postprocessed_accuracy'].append(1 - util.error_rate(y_test, preds_fair))
            results_fair['original_metric'].append(util.delta_sp(preds, sensitive_test, n_classes, n_groups))
            results_fair['postprocessed_metric'].append(util.delta_sp(preds_fair, sensitive_test, n_classes, n_groups))
            
        # Save results to a pickle file
        outfile = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/post_{dname}_{l}_{depth}_{metric}.p"

        with open(outfile, "wb") as f:
            pickle.dump(results_fair, f, protocol=pickle.DEFAULT_PROTOCOL)
            
        # Optionally return the results if needed later
        print("original_accuracy:", results_fair['original_accuracy'], "original_metric:", results_fair['original_metric'])
        print("processed_accuracy:", results_fair['postprocessed_accuracy'], "processed_metric:", results_fair['postprocessed_metric'])
        return results_fair

    elif metric == "eopp":
        alphas = [0.001, 0.01, 0.1, 0.5, 1000]
        for alpha in alphas:
            postprocessor_eopp = postprocess.PostProcessor(
                n_classes,
                n_groups,
                pred_ay_fn=predict_ay,
                criterion='eopp',
                alpha=alpha, # Fairness tolerance
            )

            postprocessor_eopp.fit(inputs_test, solver=None)

            # Evaluate
            preds = np.argmax(predict_y(X_test), axis=1)
            preds_fair = postprocessor_eopp.predict(X_test)

            
            # Save results
            results_fair['original_accuracy'].append(1 - util.error_rate(y_test, preds))
            results_fair['postprocessed_accuracy'].append(1 - util.error_rate(y_test, preds_fair))
            results_fair['original_metric'].append(util.delta_eopp(y_test, preds, sensitive_test, n_classes, n_groups))
            results_fair['postprocessed_metric'].append(util.delta_eopp(y_test, preds_fair, sensitive_test, n_classes, n_groups))

        # Save results to a pickle file
        outfile = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/post_{dname}_{l}_{depth}_{metric}.p"

        with open(outfile, "wb") as f:
            pickle.dump(results_fair, f, protocol=pickle.DEFAULT_PROTOCOL)
            
        # Optionally return the results if needed later
        print("original_accuracy:", results_fair['original_accuracy'], "original_metric:", results_fair['original_metric'])
        print("processed_accuracy:", results_fair['postprocessed_accuracy'], "processed_metric:", results_fair['postprocessed_metric'])
        return results_fair

    elif metric == "eodds":
        alphas = [0.001, 0.01, 0.1, 0.5, 1000]
        for alpha in alphas:
            postprocessor_eo = postprocess.PostProcessor(
            n_classes,
            n_groups,
            pred_ay_fn=predict_ay,
            criterion='eo',
            alpha=alpha,
            )
            postprocessor_eo.fit(inputs_test, solver=None)

            # Evaluate
            preds = np.argmax(predict_y(X_test), axis=1)
            preds_fair = postprocessor_eo.predict(X_test)

            # Save results
            results_fair['original_accuracy'].append(1 - util.error_rate(y_test, preds))
            results_fair['postprocessed_accuracy'].append(1 - util.error_rate(y_test, preds_fair))
            results_fair['original_metric'].append(util.delta_eo(y_test, preds, sensitive_test, n_classes, n_groups))
            results_fair['postprocessed_metric'].append(util.delta_eo(y_test, preds_fair, sensitive_test, n_classes, n_groups))

        # Save results to a pickle file
        outfile = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/post_{dname}_{l}_{depth}_{metric}.p"

        with open(outfile, "wb") as f:
            pickle.dump(results_fair, f, protocol=pickle.DEFAULT_PROTOCOL)
            
        # Optionally return the results if needed later
        print("original_accuracy:", results_fair['original_accuracy'], "original_metric:", results_fair['original_metric'])
        print("processed_accuracy:", results_fair['postprocessed_accuracy'], "processed_metric:", results_fair['postprocessed_metric'])
        return results_fair

    

    


