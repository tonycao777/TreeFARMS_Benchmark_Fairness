import numpy as np
import pandas as pd
import pickle
import os
import time
from utils import *
from plot import *
from fair_rset import *
from sklearn.metrics import accuracy_score
from tree_classifier import Tree
from fairoct import *
from threshold_guess import *
from cross_val import *
from post import *
from post_test import *
from make_table import *
import loader
from result import get_result, get_results

# dnames = ["adult"]
# for dname in dnames:
#     for depth in [5,6]:
#         for lamb in [0.005]: #, 0.002, 0.001]:
#             for eps in [0.05]:#[0.01, 0.02, 0.05]:
#                 print(f"{dname}_{lamb}_{depth}_{eps}")
#                 get_rset(dname, lamb, depth, eps, guess=True)

#                 plot_fairness_scatter(dname, lamb, eps)
#                 print("Generated fairness scatter plot.")
#                 filepath = "results_rset/rset_{}_guess_{}_{}_{}.p".format(dname, lamb, depth, eps)
#                 if not os.path.exists(filepath):
#                     continue
#                 get_rset_fair_results(dname, lamb, depth, eps, guess=True)
#                 for metric in ["dp", "eopp", "eodds"]:
#                     plot_compare_fairness_depth(dname, metric, lamb, depth, eps, guess=True, deltas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
"""
plot_fairness_density("compas-recid", "dp", 0.01, 4, 0.05, guess=False)

dnames = ["bank", "compas-recid", "oulad", "student-mat", "student-por", "german-credit"]
get_rset("compas-recid", 0.01, 4, 0.05, guess = False)


for dname in ["compas-recid"]:
    for depth in [1, 2, 3, 4, 5]:
        for lamb in [0.01, 0.005]: #, 0.002, 0.001]:
            for eps in [0.05]:#[0.01, 0.02, 0.05]:
                print(f"{dname}_{lamb}_{depth}_{eps}")
                get_rset(dname, lamb, depth, eps, guess=False)
                filepath = "/home/users/dc460/TreeFARMSBenchmark/results_rset/rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
                if not os.path.exists(filepath):
                    print("it failed")
                    continue
                get_rset_fair_results(dname, lamb, depth, eps, guess=False)
                for metric in ["dp", "eopp", "eodds"]:
                    plot_fairness_density(dname, metric, lamb, depth, eps, guess=False)



#Cross validation
dnames = [ "student-por"]
for dname in dnames:
    cross_validation(dname, 0.01)




# make plots

dnames = ["adult", "bank", "compas-recid", "german-credit", "oulad", "student-mat", "student-por"]
for dname in dnames:
    for lamb in [0.01, 0.005]:
        for depth in [2, 3, 4]:
            for eps in [0.05]:
                # plot_fairness_scatter(dname, lamb, eps)
                for metric in ["dp", "eopp", "eodds"]:
                    plot_fairness_density(dname, metric, lamb, depth, eps, guess=False)
                    plot_compare_fairness_depth_density(dname, metric, lamb, depth, eps, guess=False)
                    #plot_compare_fairness_depth_density(dname, metric, lamb, depth, eps, guess=False)
"""
"""
#train fairoct
## fairoct is usually very slow. It seems that depth 2 is also hard.        
dnames = ["adult", "bank", "compas-recid", "german-credit", "oulad", "student-mat", "student-por"]
metrics = ["dp", "eopp", "eodds"]
lambs = [0.01]
depths = [2]
deltas = [0.01]
for dname in dnames:
    for lamb in lambs:
        for depth in depths:
            for delta in deltas:
                for metric in metrics:
                    train_fairoct(dname, metric, lamb, depth, delta)



dnames = ["adult", "bank", "compas-recid", "german-credit", "oulad", "student-mat", "student-por"]
metrics = ["dp", "eopp", "eodds"]
lambs = [0.01]
depths = [3]
deltas = [0.01]


for dname in dnames:
    for depth in depths:
        for delta in deltas:
            for metric in metrics:
                post_test(dname, depth, 0.01, metric)

                
dnames = ["adult", "bank", "census-income", "communities", "compas-recid", "german-credit", "oulad", "student-mat", "student-por"]
for dname in dnames:
    get_dpf_results("adult", depths=[2,3,4,5], deltas=[0.01])
    post(dname, 3, 0.01, "dp")
    post(dname, 3, 0.01, "eodds")
    post(dname, 3, 0.01, "eopp")


get_result(3, "bank")
"""
final_result = get_results(2)

save_pretty_table(final_result)
