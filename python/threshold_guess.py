import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class threshold_guess:
    
    def __init__(self, max_depth, n_estimators, learning_rate, backselect, random_seed=42):
        self.d = max_depth
        self.n_est = n_estimators
        self.lr = learning_rate
        self.backselect = backselect
        self.seed = random_seed
        
    def fit_gbdt(self, X, y):
        clf = GradientBoostingClassifier(max_depth=self.d, 
                                         n_estimators=self.n_est, 
                                         learning_rate=self.lr, 
                                         random_state=self.seed)
        clf.fit(X, y)
        out = clf.score(X,y)
        return clf, out
        
    def fit(self, X, y):
        clf, acc = self.fit_gbdt(X, y)
        
        thresholds = set()
        for est in clf.estimators_:
            tree = est[0].tree_
            f = tree.feature
            t = tree.threshold
            thresholds.update([(f[i], t[i]) for i in range(len(f)) if f[i] >= 0])
        
        self.thresholds = list(thresholds)
        
        if self.backselect:
            X_new = self.transform(X)
            clf, acc_init = self.fit_gbdt(X_new, y)
            
            X_init = X_new.copy()
                
            for i in range(X_init.shape[1]-1):
                vi = clf.feature_importances_
                if vi.size > 0:
                    vi_idx = np.argmin(vi)
                    X_init = X_init.drop(X_init.columns[vi_idx], axis=1)
                    clf, acc = self.fit_gbdt(X_init, y)
                    if acc >= acc_init:
                        del self.thresholds[vi_idx]
                    else:
                        break
                else:
                    break
        
        self.thresholds.sort(key=lambda x: (x[0], x[1]))
        
        return self
    

     
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            feature_names_in = X.columns.astype(str)  # If X is a DataFrame, get the column names
        else:
            # If X is a NumPy ndarray, create fake feature names (e.g., feature_0, feature_1, ...)
            feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        # If X is not a NumPy ndarray, convert it into one
        if not isinstance(X, np.ndarray):
            X = X.values  # Convert to NumPy array
        
        # check or transform X, y into ndarrays
        if not isinstance(X, np.ndarray):
            X = X.values
        
        feature_names_out = []
        X_new = np.zeros((X.shape[0], len(self.thresholds)))
        for i in range(len(self.thresholds)):
            f, t = self.thresholds[i] 
            # check if the original column is binary
            if np.array_equal(X[:,f], X[:,f].astype(bool)):
                X_new[:, i] = X[:,f]
                feature_names_out.append(feature_names_in[f])
            else:
                X_new[X[:,f] <= t, i] = 1
                feature_names_out.append(f"{feature_names_in[f]} <= {t}")
        
        
        return pd.DataFrame(X_new, columns=feature_names_out, dtype=int)


