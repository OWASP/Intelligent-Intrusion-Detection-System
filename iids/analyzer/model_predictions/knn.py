
import numpy as np
import pandas as pd 
import os
print(os.listdir("/"))

df = pd.read_csv('/mnt/d/GSOC/Intelligent-Intrusion-Detection-System/datasets/kddcup.data.gz')
##Change according to the directory of the cloned repo w.r.t dataset location.

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


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service']= le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

X = df.iloc[:,:41]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = 0.4)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# n_neighbors are set to 3, after getting max accuracy
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

score = knn_model.score(X_test, y_test)
print(f'The score by KNClassifier is {score}')
