import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from utils import *
import os
import seaborn as sns

def plot_fairness_scatter(dname, lamb, eps):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    rset_results = np.array([]).reshape(0, 13)
    for depth in [1,2,3,4,5,6]:
        filepath = "/home/users/dc460/TreeFARMSBenchmark/results_fair/rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
        if os.path.exists(filepath):
            res = load_file(filepath)
            rset_results = np.r_[rset_results, res["results"]]

    
    rset_names = np.array(["tree_idx", "tree_depth", "tree_nleaves", 
                           "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    for i, t in enumerate(["train", "test"]):
        for j, metric in enumerate(["dp", "eopp", "eodds"]):
            axs[i,j].scatter(rset_results[:,np.where(rset_names == f"{t}_{metric}")[0][0]],
                            rset_results[:,np.where(rset_names == f"{t}_acc")[0][0]],    
                            marker="o", alpha=1, 
                            color=cm.Blues_r(rset_results[:,np.where(rset_names == "tree_depth")[0]]/8))  
            axs[i,j].set_title(f"{t} {metric}", fontsize=15)
            axs[i,j].set_xlabel(f"{t} {metric}")
            axs[i,j].set_ylabel(f"{t} acc")

    plt.tight_layout()
    plt.savefig("/home/users/dc460/TreeFARMSBenchmark/figures/{}_rset_{}_{}.png".format(dname, lamb, eps), dpi=200, bbox_inches='tight')


def plot_compare_fairness(dname, metric, lamb, eps, deltas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
    fig, axs = plt.subplots(2, 6, figsize=(20, 6))

    # get rset fairness
    rset_results = np.array([]).reshape(0, 13)
    for depth in [1,2,3,4,5,6]:
        filepath = "results_fair/rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
        if os.path.exists(filepath):
            res = load_file(filepath)
            rset_results = np.r_[rset_results, res["results"]]
    rset_names = np.array(["tree_idx", "tree_depth", "tree_nleaves", 
                           "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])

    filepath = "results_fair/dpf_{}.p".format(dname)
    res = load_file(filepath)
    dpf_results = res["results"]
    dpf_names = np.array(["config_depth","config_delta", "time", "tree_depth", "tree_nleaves", 
                           "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    
    # res = np.r_[lamb, depth, delta, train_time, fct.optim_gap, fairness_cal_time,
    #             get_fairoct_depth(fct, node_dict), get_fairoct_leaves(fct, node_dict), 
    #             train_res, test_res]
    fairoct_results = np.array([]).reshape(0, 18)
    filepath = "results_fair/fairoct_{}_{}.p".format(dname, metric)
    res = load_file(filepath)
    for k in res.keys():
        if "fair" in k:
            fairoct_results = np.r_[fairoct_results, res[k].reshape(1,-1)]
    fairoct_names = np.array(["lamb", "depth", "delta", "train_time", "opt_gap", "calc_time", 
                             "tree_depth", "tree_nleaves", "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    
    
    for j, delta in enumerate(deltas):
        rset_results_delta = rset_results[rset_results[:, np.where(rset_names == f"train_{metric}")[0][0]] <= delta]
        dpf_results_delta = dpf_results[dpf_results[:, np.where(dpf_names == f"train_{metric}")[0][0]] <= delta]
        fairoct_results_delta = fairoct_results[fairoct_results[:, np.where(fairoct_names == f"train_{metric}")[0][0]] <= delta]
        print(len(rset_results_delta), len(dpf_results_delta), len(fairoct_results_delta), flush=True)
        print("lamb",lamb, "eps",eps, "metric", metric, "delta", delta, flush=True)
        print(fairoct_results_delta.shape)

        for i, t in enumerate(["train", "test"]):
            if len(fairoct_results_delta) > 0:
                axs[i,j].scatter(fairoct_results_delta[:,np.where(fairoct_names == f"{t}_{metric}")[0][0]], 
                                fairoct_results_delta[:,np.where(fairoct_names == f"{t}_acc")[0][0]],     
                                marker="<", alpha=0.7,
                                color=cm.Greens_r(fairoct_results_delta[:,np.where(fairoct_names == "tree_depth")[0]]/8))

            if len(rset_results_delta)>0:
                axs[i,j].scatter(rset_results_delta[:,np.where(rset_names == f"{t}_{metric}")[0][0]],
                                rset_results_delta[:,np.where(rset_names == f"{t}_acc")[0][0]],    
                                marker="o", alpha=0.7, 
                                color=cm.Blues_r(rset_results_delta[:,np.where(rset_names == "tree_depth")[0]]/8))  
            if len(dpf_results_delta) > 0:
                axs[i,j].scatter(dpf_results_delta[:,np.where(dpf_names == f"{t}_{metric}")[0][0]], 
                                dpf_results_delta[:,np.where(dpf_names == f"{t}_acc")[0][0]],     
                                marker="+", alpha=0.7,
                                color=cm.Reds_r(dpf_results_delta[:,np.where(dpf_names == "tree_depth")[0]]/8))
            
            axs[i,j].set_title(f"{t} {metric} <= {delta}", fontsize=15)
            axs[i,j].set_xlabel(f"{t} {metric}")
            axs[i,j].set_ylabel(f"{t} acc")

    plt.tight_layout()
    plt.savefig("figures/{}_{}_{}_{}.png".format(dname, metric, lamb, eps), dpi=200, bbox_inches='tight')


def plot_compare_fairness_depth(dname, metric, lamb, depth, eps, guess=False, deltas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
    fig, axs = plt.subplots(2, 6, figsize=(20, 6))

    # get rset fairness
    # rset_results = np.array([]).reshape(0, 13)
    if guess:
        filepath = "/home/users/dc460/TreeFARMSBenchmark/results_fair/rset_{}_guess_{}_{}_{}.p".format(dname, lamb, depth, eps)
    else:
        filepath = "/home/users/dc460/TreeFARMSBenchmark/results_fair/rset_{}_{}_{}_{}.p".format(dname, lamb, depth, eps)
    if os.path.exists(filepath):
        res = load_file(filepath)
        rset_results = res["results"]
    else:
        return
    rset_names = np.array(["tree_idx", "tree_depth", "tree_nleaves", 
                           "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])

    filepath = "/home/users/dc460/TreeFARMSBenchmark/results_fair/dpf_{}.p".format(dname)
    res = load_file(filepath)
    dpf_results = res["results"]
    dpf_names = np.array(["config_depth","config_delta", "time", "tree_depth", "tree_nleaves", 
                           "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                           "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    dpf_results = dpf_results[dpf_results[:, np.where(dpf_names == f"config_depth")[0][0]] == depth-1]
        
    
    # fairoct_results = np.array([]).reshape(0, 18)
    # filepath = "results_fair/fairoct_{}_{}.p".format(dname, metric)
    # res = load_file(filepath)
    # for k in res.keys():
    #     if "fair" in k:
    #         fairoct_results = np.r_[fairoct_results, res[k].reshape(1,-1)]
    # fairoct_names = np.array(["lamb", "depth", "delta", "train_time", "opt_gap", "calc_time", 
    #                          "tree_depth", "tree_nleaves", "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
    #                        "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    
    
    for j, delta in enumerate(deltas):
        rset_results_delta = rset_results[rset_results[:, np.where(rset_names == f"train_{metric}")[0][0]] <= delta]
        dpf_results_delta = dpf_results[dpf_results[:, np.where(dpf_names == f"train_{metric}")[0][0]] <= delta]
        # fairoct_results_delta = fairoct_results[fairoct_results[:, np.where(fairoct_names == f"train_{metric}")[0][0]] <= delta]
        print(len(rset_results_delta), len(dpf_results_delta), flush=True)# len(fairoct_results_delta), flush=True)
        print("lamb",lamb, "eps",eps, "metric", metric, "delta", delta, flush=True)
        # print(fairoct_results_delta.shape)

        for i, t in enumerate(["train", "test"]):
            # if len(fairoct_results_delta) > 0:
            #     axs[i,j].scatter(fairoct_results_delta[:,np.where(fairoct_names == f"{t}_{metric}")[0][0]], 
            #                     fairoct_results_delta[:,np.where(fairoct_names == f"{t}_acc")[0][0]],     
            #                     marker="<", alpha=0.7,
            #                     color=cm.Greens_r(fairoct_results_delta[:,np.where(fairoct_names == "tree_depth")[0]]/8))

            if len(rset_results_delta)>0:
                axs[i,j].scatter(rset_results_delta[:,np.where(rset_names == f"{t}_{metric}")[0][0]],
                                rset_results_delta[:,np.where(rset_names == f"{t}_acc")[0][0]],    
                                marker="o", alpha=0.7, 
                                color=cm.Blues_r(rset_results_delta[:,np.where(rset_names == "tree_depth")[0]]/8))  
            if len(dpf_results_delta) > 0:
                axs[i,j].scatter(dpf_results_delta[:,np.where(dpf_names == f"{t}_{metric}")[0][0]], 
                                dpf_results_delta[:,np.where(dpf_names == f"{t}_acc")[0][0]],     
                                marker="+", alpha=0.7,
                                color=cm.Reds_r(dpf_results_delta[:,np.where(dpf_names == "tree_depth")[0]]/8))
            
            axs[i,j].set_title(f"{t} {metric} <= {delta}", fontsize=16)
            axs[i,j].set_xlabel(f"{t} {metric}", fontsize=14)
            axs[i,j].set_ylabel(f"{t} acc", fontsize=14)

    fig.suptitle("depth={}".format(depth-1), fontsize=16)
    plt.tight_layout()
    plt.savefig("/home/users/dc460/TreeFARMSBenchmark/figures/{}_{}_{}_{}_{}_{}.png".format(dname, guess, metric, lamb, depth, eps), dpi=200, bbox_inches='tight')

def load_results(dname, lamb, depth, eps, metric, guess=False):
    # Your implementation for loading results
    print("used load_result")
    if guess:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/rset_{dname}_guess_{lamb}_{depth}_{eps}.p"
    else:
        rset_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/rset_{dname}_{lamb}_{depth}_{eps}.p"
    
    if os.path.exists(rset_filepath):
        res = load_file(rset_filepath)
        rset_results = res["results"]
    else:
        raise FileNotFoundError(f"Results file {rset_filepath} not found.")

    rset_names = np.array([
        "tree_idx", "tree_depth", "tree_nleaves",
        "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
        "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"
    ])

    dpf_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/dpf_{dname}.p"
    dpf_res = load_file(dpf_filepath)
    dpf_results = dpf_res["results"]
    dpf_names = np.array([
        "config_depth", "config_delta", "time", "tree_depth", "tree_nleaves",
        "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
        "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"
    ])

    dpf_results = dpf_results[dpf_results[:, np.where(dpf_names == "config_depth")[0][0]] == depth - 1]

    if (depth == 2):
        fairoct_results = np.array([]).reshape(0, 18)
        filepath = "/home/users/dc460/TreeFARMSBenchmark/results_fair/fairoct_{}_{}.p".format(dname, "dp")
        res = load_file(filepath)
        fairoct_names = np.array(["lamb", "depth", "delta", "train_time", "opt_gap", "calc_time", 
                                "tree_depth", "tree_nleaves", "train_acc", "train_dp", "train_eopp", "train_pe", "train_eodds",
                            "test_acc", "test_dp", "test_eopp", "test_pe", "test_eodds"])
    else:
        fairoct_results = None
        fairoct_names = None

    for k in res.keys():
        if "fair" in k:
            result = res[k].reshape(1, -1)
            tree_depth = result[0, fairoct_names == "depth"]  # Get depth from fairoct_names
            if tree_depth == depth:  # Only keep the results where depth matches
                fairoct_results = np.r_[fairoct_results, result]
    
    post_filepath = f"/home/users/dc460/TreeFARMSBenchmark/results_fair/post_{dname}_{lamb}_{depth}_{metric}.p"  # Adjust for other metrics
    if os.path.exists(post_filepath):
        post_results = load_file(post_filepath)
        print(post_results)
        post_names = np.array([
            "original_accuracy", "postprocessed_accuracy", "original_metric", "postprocessed_metric"
        ])
        print("the postfile was not empty")
    else:
        post_results = None
        post_names = None
        print(f"Post-processing results file {post_filepath} not found.")
    
    
    return rset_results, rset_names, dpf_results, dpf_names, fairoct_results, fairoct_names, post_results, post_names

def plot_fairness_density(dname, metric, lamb, depth, eps, guess=False):
    plt.figure(figsize=(10,6))

    # Load the results
    rset_results, rset_names, dpf_results, dpf_names, fairoct_results, fairoct_names, post_results, post_names = load_results(dname, lamb, depth, eps, metric, guess)
    # Get the fairness results
    rset_metric = rset_results[:, np.where(rset_names == f"train_{metric}")[0][0]]
    dpf_metric = dpf_results[:, np.where(dpf_names == f"train_{metric}")[0][0]]
    if fairoct_results is None:
        fairoct_metric = None
    else:
        fairoct_metric = fairoct_results[:, np.where(fairoct_names == f"train_{metric}")[0][0]]

    # Plot density for RSET results
    sns.kdeplot(
        x=rset_metric,
        color='blue',
        label='RSET Trees',
        fill=True,
        alpha=0.5,
        thresh=0,
        common_norm=False
    )
    
    dpf_values, dpf_counts = np.unique(dpf_metric, return_counts=True)

    # Plot individual vertical lines for DPF results with heights based on the density at each x point
    plt.vlines(
        dpf_values, 
        ymin=0, 
        ymax=dpf_counts * 30,  # Set ymax to the density value at each point
        color='red', 
        linewidth=2,  # Increase line width for visibility
        label='DPF Trees',
        alpha=0.5
    )
    # Plot density for Fairoct results
    if fairoct_results is not None:
        fairoct_values, fairoct_counts = np.unique(fairoct_metric, return_counts=True)
        print(fairoct_values)

        plt.vlines(
            fairoct_values, 
            ymin=0, 
            ymax=fairoct_counts * 40,
            color='green', 
            linewidth=2, 
            label='Fairoct Trees',
            alpha=0.7
        )

    # Plot post-processed results
    if post_results is not None:
        print("it got here")
        post_metric = post_results["postprocessed_metric"]
        post_values, post_counts = np.unique(post_metric, return_counts=True)
        
        # Plot individual vertical lines for post-processed results
        plt.vlines(
            post_values, 
            ymin=0, 
            ymax=post_counts * 500,  # Adjust scale as needed
            color='purple', 
            linewidth=2, 
            label='Post-processed Trees',
            alpha=0.7
        )

    # Adding titles and labels
    plt.title(f'Density of {metric} for RSET, DPF, and Fairoct Trees (Depth={depth})', fontsize=16)
    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    print("Plotting complete")
    plt.savefig(f"/home/users/dc460/TreeFARMSBenchmark/plots/{dname}_density_{guess}_{metric}_{lamb}_{depth}_{eps}.png", dpi=200, bbox_inches='tight')
    plt.show()


def plot_compare_fairness_depth_density(dname, metric, lamb, depth, eps, guess=False):
    # Create a figure with 2 rows (training vs test) and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Load the results (assumed to be in the format of (results, names))
    rset_results, rset_names, dpf_results, dpf_names, fairoct_results, fairoct_names, post_results, post_names = load_results(dname, lamb, depth, eps, metric, guess)

    # Extract the fairness and accuracy metrics for RSET and DPF results
    rset_metric_idx = np.where(rset_names == f"train_{metric}")[0][0]
    rset_test_metric_idx = np.where(rset_names == f"test_{metric}")[0][0]
    dpf_metric_idx = np.where(dpf_names == f"train_{metric}")[0][0]
    dpf_test_metric_idx = np.where(dpf_names == f"test_{metric}")[0][0]
    rset_acc_idx = np.where(rset_names == "train_acc")[0][0]
    rset_test_acc_idx = np.where(rset_names == "test_acc")[0][0]
    dpf_acc_idx = np.where(dpf_names == "train_acc")[0][0]
    dpf_test_acc_idx = np.where(dpf_names == "test_acc")[0][0]

    # Extract x and y values for the metrics (train and test)
    rset_train_x = rset_results[:, rset_metric_idx]
    rset_train_y = rset_results[:, rset_acc_idx]
    rset_test_x = rset_results[:, rset_test_metric_idx]
    rset_test_y = rset_results[:, rset_test_acc_idx]
    
    dpf_train_x = dpf_results[:, dpf_metric_idx]
    dpf_train_y = dpf_results[:, dpf_acc_idx]
    dpf_test_x = dpf_results[:, dpf_test_metric_idx]
    dpf_test_y = dpf_results[:, dpf_test_acc_idx]
    
    if fairoct_results is not None:
        if fairoct_results.shape[0] > 0:
            fairoct_train_x = fairoct_results[:, fairoct_names == f"train_{metric}"][0]  # Take the first row (or modify as needed)
            fairoct_train_y = fairoct_results[:, fairoct_names == "train_acc"][0]
            fairoct_test_x = fairoct_results[:, fairoct_names == f"test_{metric}"][0]
            fairoct_test_y = fairoct_results[:, fairoct_names == "test_acc"][0]
        else:
            # If there are no fairOCT results, handle this gracefully
            fairoct_train_x = fairoct_train_y = fairoct_test_x = fairoct_test_y = None
    else:
        fairoct_train_x = None
        fairoct_train_y = None
        fairoct_test_x = None
        fairoct_test_y = None

    if post_results is not None:
        post_x = post_results["postprocessed_metric"]
        post_y = post_results["postprocessed_accuracy"]
    else:
        post_x = None
        post_y = None

    # Define a helper function to plot the density and scatter
    def plot_rset_dpf_fairoct(ax, rset_x, rset_y, dpf_x, dpf_y, post_x, post_y, title, fairoct_x = None, fairoct_y = None,):
        # Create the contour plot for RSET results (density estimate)
        sns.kdeplot(
            x=rset_x, 
            y=rset_y, 
            ax=ax, 
            cmap='Blues', 
            fill=True, 
            thresh=0, 
            levels=20, 
            alpha=0.4
        )

        # Create a 2D histogram to get the exact number of trees per bin (for color bar)
        heatmap, xedges, yedges = np.histogram2d(rset_x, rset_y, bins=50)

        # Plot the histogram as a heatmap (showing exact count of trees)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Purples', aspect='auto', alpha=0.7)

        # Add color bar for the exact number of trees (not just density)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Number of Trees', fontsize=12)

        # Plot the DPF results as a scatter plot (individual points)
        ax.scatter(
            dpf_x, 
            dpf_y, 
            color='red', 
            label='DPF Trees', 
            alpha=0.6, 
            edgecolors='black', 
            s=50
        )

        # Plot the FAIROCT results as a scatter plot (single point or small number of points)
        if fairoct_x is not None and fairoct_y is not None:
            ax.scatter(
                fairoct_x, 
                fairoct_y, 
                color='green', 
                label='FAIROCT Trees', 
                alpha=1.0, 
                edgecolors='black', 
                s=100, 
                marker='X'
            )

            # Plot the FAIROCT results as a scatter plot (single point or small number of points)
        if post_x is not None and post_y is not None:
            ax.scatter(
                post_x, 
                post_y, 
                color='orange', 
                label='Post Processed Trees', 
                alpha=1.0, 
                edgecolors='black', 
                s=100, 
                marker='X'
            )

        # Set the titles and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"{metric}", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)

        # Add legend
        ax.legend()

    # Plot the RSET, DPF, and FAIROCT for training data
    plot_rset_dpf_fairoct(axs[0], rset_train_x, rset_train_y, dpf_train_x, dpf_train_y, post_x, post_y, "Training: RSET, DPF, FAIROCT, and Post-Proceesed Trees", fairoct_train_x, fairoct_train_y)

    # Plot the RSET, DPF, and FAIROCT for test data
    plot_rset_dpf_fairoct(axs[1], rset_test_x, rset_test_y, dpf_test_x, dpf_test_y, post_x, post_y, "Test: RSET, DPF, FAIROCT, and Post-Proceesed Trees", fairoct_train_x, fairoct_train_y)

    # Adjust layout for better presentation
    plt.tight_layout()

    print("it got here")
    # Save the figure and show the plot
    plt.savefig(f"/home/users/dc460/TreeFARMSBenchmark/plots/contour_{dname}_{guess}_{metric}_{lamb}_{depth}_{eps}.png", dpi=200, bbox_inches='tight')
    plt.show()