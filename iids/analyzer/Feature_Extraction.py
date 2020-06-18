#importing libraries
import numpy as np
import pandas as pd 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
print(os.listdir("/"))

#change it according ur directory
df = pd.read_csv('/Intelligent-Intrusion-Detection-System/datasets/kddcup.data.gz')
print(df.shape())

df.columns =["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

X = df.iloc[:,:41]
y = df.iloc[:,-1]
X.head()

#Pre-processing of categorial data columns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service']= le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

#Splitting of data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = 0.3) 

#Scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

#U can easily change the model by just replacing here
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

rfecv = RFECV(estimator=model1, step=1, cv=StratifiedKFold(2), scoring='accuracy' )
rfecv.fit(X_train_scaled,y_train)


print('Optimal number of features: {}'.format(rfecv.n_features_))

#Feature Selection plot
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()

dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)

#Feature Ranking plot
plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

