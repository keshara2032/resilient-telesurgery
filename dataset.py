from typing import List
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
    
    def __init__(self, files_path: List[str], observation_window_size: int, prediction_window_size: int, step: int = 0, onehot = False):
        
        self.files_path = files_path
        self.observation_window_size = observation_window_size
        self.prediction_window_size = prediction_window_size
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
        _X = self.X[trial_start:trial_end]
        _Y = self.Y[trial_start:trial_end]
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
        Y = torch.from_numpy(Y).to(torch.float32).to(self.device)
        return X, Y

        
    def _load_data(self):
        X = []
        X_image = []
        Y = []
        self.samples_per_trial = []
        
        for kinematics_path, video_path in self.files_path:
            if os.path.isfile(kinematics_path) and kinematics_path.endswith('.csv'):
                kinematics_data = pd.read_csv(kinematics_path)

                kin_data = kinematics_data.iloc[:,:-1]
                kin_label = kinematics_data.iloc[:,-1]

                X_image.append(pd.DataFrame({'file_name': [video_path]*len(kin_data), 'frame_number': np.arange(len(kin_data))}))
                X.append(kin_data.values)
                Y.append(kin_label.values)
                self.samples_per_trial.append(len(kin_data))

        self.feature_names = kinematics_data.columns.to_list()[:-1]
        self.max_len = max([d.shape[0] for d in X])
        
        # label encoding and transformation
        self.le.fit(Y[0])
        Y = [self.le.transform(yi) for yi in Y]
        self.target_names = self.le.classes_

        # one-hot encoding
        Y = [yi.reshape(len(yi), 1) for yi in Y]
        Y = [self.enc.fit_transform(yi) for yi in Y]
        
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
    def collate_fn(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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

        x_batch = torch.from_numpy(X)
        # xi_batch = torch.from_numpy(X_image)
        y_batch = torch.from_numpy(Y)
        yy_batch = torch.from_numpy(Future_Y)
        p_batch = torch.from_numpy(P)
        
        x_batch = x_batch.to(torch.float32)
        # xi_batch = xi_batch.to(torch.float32)
        y_batch = y_batch.to(torch.float32)
        yy_batch = yy_batch.to(torch.float32)
        p_batch = p_batch.to(torch.float32)

        x_batch = x_batch.to(device)
        # xi_batch = xi_batch.to(device)
        y_batch = y_batch.to(device)
        yy_batch = yy_batch.to(device)
        p_batch = p_batch.to(device)

        return (x_batch, None, y_batch, yy_batch, p_batch)


def get_dataloaders(task: str,
                    subject_id_to_exclude: str,
                    observation_window: int,
                    prediction_window: int,
                    batch_size: int):
    
    from typing import List
    import os
    from functools import partial
    import torch

    from torch.utils.data import DataLoader
    from dataset import LOUO_Dataset
    from datagen import tasks
    

    def _get_files_except_user(task, data_path, subject_id_to_exclude: int) -> List[str]:
        assert(task in tasks)
        files = os.listdir(data_path)
        csv_files = [p for p in files if p.endswith(".csv")]
        with open(os.path.join(data_path, "video_files.txt"), 'r') as fp:
            video_files = fp.read().strip().split('\n')
        csv_files.sort(key = lambda x: os.path.basename(x)[:-4])
        video_files.sort(key = lambda x: os.path.basename(x)[:-4])
        except_user = [(os.path.join(data_path, p), v) for (p, v) in zip(csv_files, video_files) if not p.startswith(f"{task}_S0{subject_id_to_exclude}")]
        user = [(os.path.join(data_path, p), v) for (p, v) in zip(csv_files, video_files) if p.startswith(f"{task}_S0{subject_id_to_exclude}")]
        return except_user, user 


    # building train and validation datasets and dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("ProcessedDatasets", task)
    train_files_path, valid_files_path = _get_files_except_user(task, data_path, subject_id_to_exclude)
    train_dataset = LOUO_Dataset(train_files_path, observation_window, prediction_window, onehot=True)
    valid_dataset = LOUO_Dataset(valid_files_path, observation_window, prediction_window, onehot=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device))
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device)) 

    return train_dataloader, valid_dataloader                  