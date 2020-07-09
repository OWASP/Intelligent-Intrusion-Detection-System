from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
#loading our trained model
print(" Model loading.......")
model = load_model('attack_labe.hdf5') #after training #TODO
print("Model loaded!!")


#class Ml_Algo and functions 

class Random_Forest_Classifier():
    def __init__(self):
        default_path = "//"
        self.model = load_model("")
      
    def preprocessing(self, input_data):
        df = pd.read_csv('input_data')
        df.columns =["duration","protocol_type","service","flag","src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        le = LabelEncoder()
        df['protocol_type'] = le.fit_transform(df['protocol_type'])
        df['service']= le.fit_transform(df['service'])
        df['flag'] = le.fit_transform(df['flag'])
        
        return input_data

    def prediction(self):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

            return prediction
