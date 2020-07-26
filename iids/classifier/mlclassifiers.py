#import all the modules below
from sklearn import tree




class DecisionTree:

    def __init__(self, data, features_list="all", criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]





    def train(self):        
        
