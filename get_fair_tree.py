import subprocess
import pandas as pd
import json
import time
#from tree_classifier import TreeClassifier # Import the tree classification model
def run_dpf(dname, depth, delta):
    #subprocess.call(['./gosdt', 'temp.csv', 'temp-config.json'])
    dfpath = "/home/users/dc460/TreeFARMSBenchmark/dpf/data/{}-train-binarized.csv".format(dname)
    outpath = "/home/users/dc460/TreeFARMSBenchmark/dpf/build/trees/{}-{}-{}.json".format(dname, depth, delta)
    subprocess.call(['/home/users/dc460/TreeFARMSBenchmark/dpf/build/DPF', '-file',  dfpath, '-mode', 'best', "-max-depth", str(depth), "-max-num-nodes", str(2**depth-1), "-outfile", outpath, "-stat-test-value", str(delta)])
# -stat-test-value = 0.01
# ./DPF -file ../data/german-credit-binarized.csv -mode best -max-depth 1 -max-num-nodes 1 -outfile german-credit-1.json

"""
dnames = ["bank", "oulad", "student-mat", "student-por"]
for dname in dnames:
    print(dname, flush=True)
    for depth in [1,2,3,4,5]:
        for delta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
            print(f"depth:{depth}, delta:{delta}", flush=True)
            s_time = time.time()
            run_dpf(dname, depth, delta)
            training_time = time.time()-s_time
            print("training time:", training_time)
  
            with open("rsult_summary", 'a+') as f:
                f.write(';'.join([dname, str(depth), str(delta), str(training_time)]) + '\n')
"""