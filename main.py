import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from math import floor
from itertools import product
from sklearn import preprocessing
from lightgbm import LGBMRegressor


class Node:
    def __init__(self):
        self.left = None            # node
        self.right = None           # node
        self.split = None           # int
        self.price = None           # float
        self.feature = None         # int
        self.revenue = None         # float
        self.instances = None       # list
        self.depth = 0              # int
        self.parent = None          # node
        self.color = 'white'        # str
        self.test_price = None      # float
        self.test_revenue = None    # float
        self.test_instances = None  # list


class StudentPrescriptionTree:
    """
    Target values are not needed, only the data. Price is put in artificially to obtain counterfactual
    purchasing probabilities, and thereby expected revenue
    """
    def __init__(self, data, model, categorical_feature_idx=None, max_depth=-1):
        # self.root = None                      # root node consists of all training instances
        self.data = data
        self.model = model                      # lightGBM, xgBoost or GLM
        self.categorical_feature_idx = categorical_feature_idx
        self.max_depth = max_depth
        self.predict_price = np.empty([0, 2])
        self.predict_revenue = []
        self.root = Node()
        self.min_instances = floor(np.sqrt(len(data)))
        self.root.instances = np.arange(len(self.data)).tolist()

        if categorical_feature_idx is None:
            self.categorical_feature_idx = []
        elif not isinstance(categorical_feature_idx, list):
            raise Exception('categorical_feature_idx must be of type list.')
        else:
            self.categorical_feature_idx = categorical_feature_idx

    def __len__(self):
        return self.size(self.root)

    def get_root(self, data):
        """
        Insert all data instances into the root node
        """
        root = Node()
        root.instances = np.arange(len(data)).tolist()  # all data instances
        return root

    def counterfactual(self, data):
        # return np.random.normal(5, 5), np.random.normal(0, 1)
        if len(data) == 0:
            max_revenue = 0
            optimal_price = np.nan

            return optimal_price, max_revenue
        low_price = self.data.price.quantile(0.05)
        high_price = self.data.price.quantile(0.95)
        #prices = np.linspace(low_price, high_price, 11)
        prices = np.linspace(1.99, 4.99, 7)

        revenue = []
        for price in prices:
            # counterfactual purchase probability
            counterfactual = data.copy()
            counterfactual.price = price
            purchase_prob = self.model.predict_proba(counterfactual)[:, 0]
            revenue.append(np.mean(price*purchase_prob))

        max_revenue = np.max(revenue)
        optimal_price = prices[np.where(revenue == max_revenue)][0]
        return optimal_price, max_revenue

    def split_tree(self, root=None):
        l_node = None
        r_node = None
        data = self.data

        data = data.iloc[root.instances]                                # get instances stored in node
        optimal_price, max_revenue = self.counterfactual(data)
        root.price = optimal_price                                      # assign optimal price for trivial partition

        for r in np.arange(len(data.columns)):
            for j in np.arange(data.iloc[:, r].nunique()):

                # continuous variable
                if r not in self.categorical_feature_idx:  # fix bug: looping over unique values only
                    if data.iloc[j, r] == data.iloc[j+1, r]:
                        continue
                    else:
                        col = data.iloc[:, r].sort_values()
                        h = (col.iloc[j] + col.iloc[j+1])/2             # split as median of contiguous distinct values

                    s1 = data[data.iloc[:, r] <= h]                     # split data
                    s2 = data[data.iloc[:, r] > h]

                # categorical variable
                else:
                    #categories = data.iloc[:, r].unique()
                    #h = categories[j]
                    h = j
                    s1 = data[data.iloc[:, r] == h]
                    s2 = data[data.iloc[:, r] != h]

                new_optimal_price1, new_max_revenue1 = self.counterfactual(s1)
                new_optimal_price2, new_max_revenue2 = self.counterfactual(s2)
                new_revenue = new_max_revenue1 + new_max_revenue2

                if new_revenue > max_revenue:
                    max_revenue = new_revenue
                    root.split = h

                    l_node = Node()
                    l_node.price = new_optimal_price1
                    l_node.revenue = new_max_revenue1
                    l_node.instances = np.where(data.iloc[:, r] == h)[0].tolist()
                    l_node.depth = root.depth + 1
                    l_node.parent = root

                    r_node = Node()
                    r_node.price = new_optimal_price2
                    r_node.revenue = new_max_revenue2
                    r_node.instances = np.where(data.iloc[:, r] != h)[0].tolist()
                    r_node.depth = root.depth + 1
                    r_node.parent = root

                    root.left = l_node
                    root.right = r_node
                    root.revenue = max_revenue
                    root.feature = r

    def not_leaf(self, node):
        a = node.depth < self.max_depth
        b = len(node.instances) >= self.min_instances
        c = node.left is None
        not_leaf = a and b and c
        return not_leaf

    def grow_tree2(self, node=None):
        if node is None:
            node = self.root
            if self.not_leaf(node):
                self.split_tree(node)
                self.grow_tree2(node.left)
            else:
                return
        else:
            if self.not_leaf(node):
                self.split_tree(node)
                self.grow_tree2(node.left)
            else:
                while node == node.parent.right:
                    node = node.parent
                    if node == self.root:
                        return
                node = node.parent.right
                self.grow_tree2(node)

    # recursively partition sample space
    def grow_tree(self, node=None):
        if node == self.root:
            return
        elif node is None:
            node = self.root
            self.split_tree(node)
            self.grow_tree(node.left)
        else:
            if node.depth < self.max_depth and len(node.instances) >= self.min_instances and node.left is None:        # not leaf and no children
                self.split_tree(node)                                                       # sprout from left node
                self.grow_tree(node.left)
            else:
                node.color = 'blue'
                node = node.parent.right
                if node.depth < self.max_depth and len(node.instances) >= self.min_instances and node.right is None:   # not leaf and no children
                    self.split_tree(node)                                                   # sprout from right node
                    self.grow_tree(node.left)
                else:
                    node.color = 'blue'
                    node = node.parent
                    self.grow_tree(node)

    def split_test_data(self, test, node):
        if node is not None:
            # split test data using node from tree
            test = test.iloc[node.test_instances, :]

            if node.feature not in self.categorical_feature_idx:
                s1 = test[test.iloc[:, node.feature] <= node.split]
                s2 = test[test.iloc[:, node.feature] > node.split]
            else:
                s1 = test[test.iloc[:, node.feature] == node.split]
                s2 = test[test.iloc[:, node.feature] != node.split]

            l_node = node.left
            optimal_price, max_revenue = self.counterfactual(s1)
            l_node.test_price = optimal_price
            l_node.test_revenue = max_revenue
            l_node.test_instances = np.where(test.iloc[:, node.feature] <= node.split)[0].tolist()

            r_node = node.right
            optimal_price, max_revenue = self.counterfactual(s2)
            r_node.test_price = optimal_price
            r_node.test_revenue = max_revenue
            r_node.test_instances = np.where(test.iloc[:, node.feature] > node.split)[0].tolist()

    def children(self, node):
        # node has children
        return node.left is not None and node.right is not None

    def fit(self, test, node=None):
        if node is None:
            node = self.root

            # get test root node attributes
            optimal_price, max_revenue = self.counterfactual(test)
            node.test_price = optimal_price
            node.test_revenue = max_revenue
            node.test_instances = np.arange(len(test)).tolist()

            self.split_test_data(test, node)
            self.fit(test, node.left)
        else:
            if self.children(node):
                self.split_test_data(test, node)
                self.fit(test, node.left)
            else:
                idx = node.test_instances
                prices = np.array([idx, [node.price]*idx.__len__()]).transpose()
                self.predict_price = np.concatenate([self.predict_price, prices], axis=0)
                self.predict_revenue.append(node.test_revenue)
                while node == node.parent.right:
                    node = node.parent
                    if node == self.root:
                        self.predict_price = np.sort(self.predict_price, axis=0)[:, 1]
                        self.predict_revenue = np.sum(self.predict_revenue)
                        return
                node = node.parent.right
                self.fit(test, node)

    def get_leaf_node(self, node, k):
        for i in range(len(k)):
            if node is None:
                break
            if k[i] == 1:
                node = node.right
            elif k[i] == -1:
                node = node.left
        return node

    def get_revenue(self, depth=0):
        if depth == 0:
            return self.root.test_revenue
        else:
            revenue = [0]
            for k in product([-1, 1], repeat=depth):
                node = self.get_leaf_node(self.root, k=k)
                if node is not None:
                    revenue.append(node.test_revenue)
            return sum(revenue)

    def print_tree(self, node, level=0):
        if node is not None:
            self.print_tree(node.left, level + 1)
            print(' ' * 4 * level + '-> ' + node.key)
            self.print_tree(node.right, level + 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # import cleaned data
    X = pd.read_csv('cleaned_strawberry_dataset.csv')

    # split data
    y = X['purchased']
    X = X.drop('purchased', axis=1)

    categorical_feature = np.arange(len(X.columns)).tolist()  # [0, 1,  2,  3,  5,  6,  7,  8,  9, 10, 11]

    num_round = 50  # number of boosting rounds as used in Biggs
    param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': '',
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'verbose': 1,
        'max_depth': -1
    }

    # use half the data for the teacher model and half for the student
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedShuffleSplit
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = StratifiedShuffleSplit(X_res, y_res, test_size=0.5, random_state=49)

    model = lgb.LGBMClassifier(n_estimators=num_round,
                               categorical_feature=np.arange(len(X.columns)))
    model.fit(X_train, y_train)

    predicted_revenue = []
    spt = StudentPrescriptionTree(X_train, model=model,
                                  categorical_feature_idx=categorical_feature, max_depth=5)

    spt.grow_tree2()
    spt.fit(X_test)

    depths = np.arange(0, 8)
    for depth in depths:
        predicted_revenue.append(spt.get_revenue(depth=depth))

    plt.plot(depths, predicted_revenue)
    plt.xlabel('Depth')
    plt.ylabel('Average Predicted Revenue')
    # print(pd.Series(spt.predict_price).describe())

