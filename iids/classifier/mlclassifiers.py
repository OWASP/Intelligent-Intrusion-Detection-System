#import all the modules below
from sklearn import tree
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import joblib
from main.management.commands import *
#import all necessay packages and all modules 
#
# remove false from save model

class DecisionTree():

    def __init__(self, data, features_list="all", criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]


    def train(self):
        
        model.fit(X_train,y_train)  
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            #return False
        return True 


class RandomForest():

    def __init__(self,data, features_list="all",criterion="gini", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, n_jobs=-1, verbose=1, ccp_alpha=0.0, max_samples=None, n_estimators=100, max_depth=2,random_state=0,class_weight='balanced'):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]

    def train(self):
        
        model.fit(X_train,y_train)  #model will be declared as global variable in views.py
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            return False
        return True 



class KNeighbors():
    def __init__(self,data, features_list="all",algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=3, p=2, weights='uniform'):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]

    def train(self):
        
        model.fit(X_train,y_train)  
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            return False
        return True 


        
class GaussianProcess():
    def __init__(self,data, features_list="all",copy_X_train=True,max_iter_predict=100, multi_class='one_vs_rest',n_jobs=None, n_restarts_optimizer=0,optimizer='fmin_l_bfgs_b', random_state=None,warm_start=False):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]


 
    def train(self):
        
        model.fit(X_train,y_train)  
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            return False
        return True 


class AdaBoost():
    def __init__(self,data, features_list="all",algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=None):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]

 
    def train(self):
        
        model.fit(X_train,y_train)  
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            return False
        return True 
 
 
class MLP():
    def __init__(self,data, features_list="all",activation='relu', alpha=1, batch_size='auto', beta_1=0.9,beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(100,), learning_rate='constant', learning_rate_init=0.001, max_fun=15000, max_iter=1000,momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam',tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False):
        self.features_list = features_list
        if self.features_list == "all":
            self.data = data
        else:
            self.data = data[features_list]
    
          
  
    def train(self):
               
        model.fit(X_train,y_train) 
        score = model.score(X_test, y_test)
        print(f'The score by {model} is {score}')
        
        return model, score
    
    def save_model(self,model,filename):
        try:
            joblib.dump(model,filename)
            print("Model saved to the disk") 
        except Exception as e:
            raise IOError("Error saving model data to disk: {}".format(str(e)))
            return False
        return True 

