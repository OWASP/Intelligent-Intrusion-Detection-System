#Import all the modules
from sklearn.preprocessing import LabelEncoder
#from main.commands.get_data import *
# pylint: disable=E1101

class Preprocessor:
    
    def __init__(self, data):
        self.data = data

    def removeNullValues(self,data):
        "This method is used to remove null values from the dataset"
        
        return self.data.dropna()

    def labelEncoder(self, columns):
        "Label Encoder that converts categorical data"

        le = LabelEncoder()
        for each in columns:
            self.data[str(each)] = le.fit_transform(df[str(each)])

        return self.data

    def train_test(self,data):
        global X_train,X_test,y_train,y_test
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = 0.4)  

        return data

    #for Neural Networks

    #"""Convert the string part of the data into numbers, and convert the input 41-dimensional features into an 8*8 matrix """
    def __encode_data(self, data_X, data_y):
        self._encoder['protocal'].fit(list(set(data_X[:, 1])))
        self._encoder['service'].fit(list(set(data_X[:, 2])))
        self._encoder['flag'].fit((list(set(data_X[:, 3]))))
        self._encoder['label'].fit(list(set(data_y)))
        data_X[:, 1] = self._encoder['protocal'].transform(data_X[:, 1])
        data_X[:, 2] = self._encoder['service'].transform(data_X[:, 2])
        data_X[:, 3] = self._encoder['flag'].transform(data_X[:, 3])
        data_X = np.pad(data_X, ((0, 0), (0, 64 - len(data_X[0]))), 'constant').reshape(-1, 1, 8, 8)
        data_y = self._encoder['label'].transform(data_y)
        return data_X, data_y    

    #"""Split the data into training and test sets, and convert to TensorDataset object"""
    def __split_data_to_tensor(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int))
        )
        return train_dataset, test_dataset

    def decode(self, data, label=False):
            if not label:
                _data = list(data)
                _data[1] = self._encoder['protocal'].inverse_transform([_data[1]])[0]
                _data[2] = self._encoder['service'].inverse_transform([_data[2]])[0]
                _data[2] = self._encoder['flag'].inverse_transform([_data[3]])[0]
                return _data
            return self._encoder['label'].inverse_transform(data)
        
    def encode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].transform([_data[1]])[0]
            _data[2] = self._encoder['service'].transform([_data[2]])[0]
            _data[3] = self._encoder['flag'].transform([_data[3]])[0]
            return _data
        return self._encoder['label'].transform([data])[0]

     