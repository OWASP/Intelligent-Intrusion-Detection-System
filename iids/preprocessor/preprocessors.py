#Import all the modules
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    
    def __init__(self, data):
        self.data = data

    def removeNullValues(self):
        "This method is used to remove null values from the dataset"
        
        return self.data.dropna()
    
    def labelEncoder(self, columns):i
        "Label Encoder that converts categorical data"

        le = LabelEncoder()
        for each in columns:
            self.data[str(each)] = le.fit_transform(df[str(each)])

        return self.data
    
    def train_test(self,data): 
            

        return data

    #for NN's
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