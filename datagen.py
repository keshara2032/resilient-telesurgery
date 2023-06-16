import numpy as np
import pandas as pd
import os

root_path = "./Datasets/dV/Peg_Transfer/"
gestures_path = os.path.join(root_path, "gestures")
kinematics_path = os.path.join(root_path, "kinematics")
os.makedirs("./new_dataset/gestures")

for file in os.listdir(gestures_path):
    if file.startswith("Peg_Transfer"):
        file_path = os.path.join(gestures_path, file)
        labels = pd.read_csv(file_path, sep=' ', index_col=None, header=None)
        labels = labels.set_axis(['Start', "Stop", "MP", "Success", "X"], axis=1).drop("X", axis=1)
        kinematics = pd.read_csv(os.path.join(kinematics_path, file[:-3] + 'csv'), index_col=None)
        kinematics['label'] = ['-'] * len(kinematics)
        for i, row in labels.iterrows():
            start, stop, mp = int(row['Start']), int(row['Stop']), row['MP']
            kinematics.loc[start:stop, 'label'] = mp
        kinematics = kinematics[kinematics['label'] != '-']
        kinematics.to_csv("./new_dataset/gestures/" + file[:-3] + 'csv')