from typing import List
import torch
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



class LOUO_Dataset(Dataset):
    
    def __init__(self, files_path: List[str], observation_window_size: int, prediction_window_size: int, step: int = 0, onehot = False):
        
        self.files_path = files_path
        self.observation_window_size = observation_window_size
        self.prediction_window_size = prediction_window_size
        self.le = preprocessing.LabelEncoder()
        self.onehot = onehot
        if onehot:
            self.enc = preprocessing.OneHotEncoder(sparse_output=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (self.X, self.Y) = self._load_data()   
        if step > 0:
            self.X, self.Y = self.X[::step], self.Y[::step] # resampling the data (e.g. in going from 30Hz to 10Hz, set step=3)   
        
    def _load_data(self):
        X = []
        Y = []
        
        for path in self.files_path:
            if os.path.isfile(path) and path.endswith('.csv'):
                kinematics_data = pd.read_csv(path)

                kin_data = kinematics_data.iloc[:,:-1]
                kin_label = kinematics_data.iloc[:,-1]
            
                X.append(kin_data.values)
                Y.append(kin_label.values)
        
        # label encoding and transformation
        self.le.fit(Y[0])
        Y = [self.le.transform(yi) for yi in Y]

        # one-hot encoding
        Y = [yi.reshape(len(yi), 1) for yi in Y]
        Y = [self.enc.fit_transform(yi) for yi in Y]
        
        # store data inside a single 2D numpy array
        X = np.concatenate(X)
        Y = np.concatenate(Y)

        return X, Y
    
    def __len__(self):
        # this should return the size of the dataset
        return self.Y.shape[0] - self.observation_window_size - self.prediction_window_size - 1
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx + 1 : idx + self.observation_window_size + 1]
        target = self.Y[idx : idx + self.observation_window_size + 1] # one additional observation is given for recursive decoding in recognition task
        pred_target = self.Y[idx + self.observation_window_size + 1 : idx + self.observation_window_size + self.prediction_window_size + 1]
        
        return features, target, pred_target
    
    @staticmethod
    def collate_fn(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        X = []
        Y = []
        P = []
        for x, y, p in batch:
            X.append(x)
            Y.append(y)
            P.append(p)
        X = np.array(X)
        Y = np.array(Y)
        P = np.array(P)

        x_batch = torch.from_numpy(X)
        y_batch = torch.from_numpy(Y)
        p_batch = torch.from_numpy(P)
        
        x_batch = x_batch.to(torch.float32)
        y_batch = y_batch.to(torch.float32)
        p_batch = p_batch.to(torch.float32)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        p_batch = p_batch.to(device)

        return (x_batch, y_batch, p_batch)                    