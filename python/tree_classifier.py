import numpy as np
from collections import deque
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class Tree:
    """
    Unified representation of a tree classifier in Python

    This class accepts a dictionary representation of a tree classifier and decodes it into an interactive object

    Additional support for encoding/decoding layer can be layers if the feature-space of the model differs from the feature space of the original data
    """
    def __init__(self, tree):
        # tree is in json format

        self.tree = tree

        children_left = []
        children_right = []
        feature = []
        prediction = []

        queue = deque([self.tree])
        while queue:
            i = len(children_left)
            node = queue[0]
            queue.popleft()
            is_leaf = "prediction" in node
            children_left.append(i if is_leaf else i+len(queue)+1)
            children_right.append(i if is_leaf else i+len(queue)+2)
            feature.append(-1 if is_leaf else node["feature"])
            prediction.append(node["prediction"] if is_leaf else -1)
            if not is_leaf:
                queue.append(node["true"])
                queue.append(node["false"])
        self.children_left = np.array(children_left)
        self.children_right = np.array(children_right)
        self.feature = np.array(feature)
        self.prediction = np.array(prediction)

    
    def predict(self, X):
        "X is (n, p) numpy array"

        n_samples = X.shape[0]
        current_node = np.zeros(n_samples, dtype=np.int32)
        while True:
            feature_indices = self.feature[current_node]
            go_left = X[np.arange(n_samples), feature_indices].astype('bool')
            current_node = np.where(go_left, self.children_left[current_node], self.children_right[current_node])
            if np.all(self.children_left[current_node] == self.children_right[current_node]):
                break
        leaf_nodes = current_node
        predictions = self.prediction[leaf_nodes]

        return predictions

    
    def score(self, X, y, weight=None):
        y_hat = self.predict(X)
        if weight == "balanced":
            return balanced_accuracy_score(y, y_hat)
        else:
            return accuracy_score(y, y_hat)



    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        leaves_counter = 0
        nodes = [self.tree]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaves_counter += 1
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaves_counter
    
    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        nodes_counter = 0
        nodes = [self.tree]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                nodes_counter += 1
            else:
                nodes_counter += 1
                nodes.append(node["true"])
                nodes.append(node["false"])
        return nodes_counter


    def maximum_depth(self, node=None):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if node is None:
            node = self.tree
        if "prediction" in node:
            return 1
        else:
            return 1 + max(self.maximum_depth(node["true"]), self.maximum_depth(node["false"]))

    