from typing import List
from functools import partial
import torch
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from torchvision.io import read_video


image_features = np.random.randn(480, 720, 3)

class LOUO_Dataset(Dataset):
    
    def __init__(self,
                files_path: List[str],
                observation_window_size: int,
                prediction_window_size: int,
                step: int = 0,
                onehot = False,
                class_names = [],
                feature_names = []):
        
        self.files_path = files_path
        self.observation_window_size = observation_window_size
        self.prediction_window_size = prediction_window_size
        self.target_names = class_names
        self.feature_names = feature_names
        self.le = preprocessing.LabelEncoder()
        self.onehot = onehot
        if onehot:
            self.enc = preprocessing.OneHotEncoder(sparse_output=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = step
        (self.X, self.X_image, self.Y) = self._load_data()   
        if step > 0:
            self.X, self.X_image, self.Y = self.X[::step], self.X_image[::step], self.Y[::step] # resampling the data (e.g. in going from 30Hz to 10Hz, set step=3)

    def get_feature_names(self):
        return self.feature_names
    
    def get_target_names(self):
        return self.target_names
    
    def get_num_trials(self):
        return len(self.samples_per_trial)
    
    def get_max_len(self):
        return self.max_len
    
    def get_trial(self, trial_id: int, window_size: int = -1):
        if trial_id == 0:
            trial_start = 0
        else:
            trial_start = sum(self.samples_per_trial[:trial_id])
        trial_end = trial_start + self.samples_per_trial[trial_id]
        _X = self.X[trial_start + 1 : trial_end + 1]
        _Y = self.Y[trial_start : trial_end + 1]
        if window_size > 0:
            X = []
            Y = []
            num_windows = _X.shape[0]//window_size
            for i in range(num_windows):
                X.append(_X[i*window_size:(i+1)*window_size])
                Y.append(_Y[i*window_size:(i+1)*window_size])
            X = np.array(X)
            Y = np.array(Y)
        else:
            X = _X
            Y = _Y
        X = torch.from_numpy(X).to(torch.float32).to(self.device) # shape [num_windows, window_size, features_dim]
        target_type = target_type = torch.float32 if self.onehot else torch.long
        Y = torch.from_numpy(Y).to(target_type).to(self.device)
        return X, Y

        
    def _load_data(self):
        X = []
        X_image = []
        Y = []
        self.samples_per_trial = []
        
        for kinematics_path, video_path in self.files_path:
            if os.path.isfile(kinematics_path) and kinematics_path.endswith('.csv'):
                kinematics_data = pd.read_csv(kinematics_path)

                feature_names = kinematics_data.columns.to_list()[:-1] if not self.feature_names else self.feature_names
                kin_data = kinematics_data.loc[:, feature_names]
                kin_label = kinematics_data.iloc[:,-1] # last column is always taken to be the target class

                X_image.append(pd.DataFrame({'file_name': [video_path]*len(kin_data), 'frame_number': np.arange(len(kin_data))}))
                X.append(kin_data.values)
                Y.append(kin_label.values)
                self.samples_per_trial.append(len(kin_data))

        
        self.max_len = max([d.shape[0] for d in X])
        
        # label encoding and transformation
        if not self.target_names:
            self.le.fit(np.concatenate(Y))
        else:
            self.le.fit(np.array(self.target_names))
        self.target_names = self.le.classes_
        print(self.target_names)
        Y = [self.le.transform(yi) for yi in Y]
        print(Y[0])

        # one-hot encoding
        if self.onehot:
            self.enc.fit(np.arange(len(self.target_names)).reshape(-1, 1))
            Y = [yi.reshape(len(yi), 1) for yi in Y]
            Y = [self.enc.transform(yi) for yi in Y]
            print(Y[0])
        
        
        # store data inside a single 2D numpy array
        X_image = pd.concat(X_image, axis=0)
        X = np.concatenate(X)
        Y = np.concatenate(Y)

        return X, X_image, Y, 
    
    def __len__(self):
        # this should return the size of the dataset
        return self.Y.shape[0] - self.observation_window_size - self.prediction_window_size - 1
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        kinematic_features = self.X[idx + 1 : idx + self.observation_window_size + 1]
        # image_features = self.read_window(idx)
        target = self.Y[idx : idx + self.observation_window_size + 1] # one additional observation is given for recursive decoding in recognition task
        gesture_pred_target = self.Y[idx + self.observation_window_size + 1 : idx + self.observation_window_size + self.prediction_window_size + 1]
        traj_pred_target = self.X[idx + self.observation_window_size + 1 : idx + self.observation_window_size + self.prediction_window_size + 1]
        
        return kinematic_features, image_features, target, gesture_pred_target, traj_pred_target
    
    def read_window(self, start_idx: int):
        video_frames = self.X_image.iloc[start_idx + 1 : start_idx + self.observation_window_size + 1]
        image_arrays = []
        prev_file_name = video_frames.iloc[0]['file_name']
        cap = cv2.VideoCapture(prev_file_name)
        for _, row in video_frames.iterrows():
            file_name, frame_number = row['file_name'], row['frame_number']
            if file_name != prev_file_name:
                cap = cv2.VideoCapture(file_name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            res, frame = cap.read()
            image_arrays.append(frame)
            prev_file_name = file_name
        image_arrays = np.array(image_arrays)
        return image_arrays

    
    @staticmethod
    def collate_fn(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), target_type=torch.float32, cast: bool = True):
        X = []
        # X_image = []
        Y = []
        Future_Y = []
        P = []
        for x, xi, y, yy, p in batch:
            X.append(x)
            # X_image.append(xi)
            Y.append(y)
            Future_Y.append(yy)
            P.append(p)
        # X_image = np.array(X_image)
        X = np.array(X)
        Y = np.array(Y)
        Future_Y = np.array(Future_Y)
        P = np.array(P)

        if cast:
            # cast to torch tensor
            x_batch = torch.from_numpy(X)
            # xi_batch = torch.from_numpy(X_image)
            y_batch = torch.from_numpy(Y)
            yy_batch = torch.from_numpy(Future_Y)
            p_batch = torch.from_numpy(P)
            
            # cast to appropriate data type
            x_batch = x_batch.to(torch.float32)
            # xi_batch = xi_batch.to(torch.float32)
            y_batch = y_batch.to(target_type)
            yy_batch = yy_batch.to(target_type)
            p_batch = p_batch.to(torch.float32)

            # cast to appropriate device
            x_batch = x_batch.to(device)
            # xi_batch = xi_batch.to(device)
            y_batch = y_batch.to(device)
            yy_batch = yy_batch.to(device)
            p_batch = p_batch.to(device)
        else:
            x_batch = X
            # xi_batch = X_image
            y_batch = Y
            yy_batch = Future_Y
            p_batch = P

        return (x_batch, None, y_batch, yy_batch, p_batch)


                