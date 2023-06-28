from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import altair as alt
from altair_saver import save
import altair_viewer
import matplotlib.pyplot as plt

from dataset import LOUO_Dataset


def get_classification_report(pred, gt, target_names):
    labels=np.arange(0,len(target_names),1)
    report = classification_report(gt, pred, target_names=target_names, labels=labels, output_dict=True)
    return pd.DataFrame(report).transpose()

def visualize_gesture_ts(pred, gt, target_names):

    def _convert_label_to_range(labels):
        df = pd.DataFrame({'gesture': labels})
        pred_index_changes = df["gesture"].diff()[df["gesture"].diff() != 0].index.values

        changes_df = df.iloc[pred_index_changes]
        changes_df.reset_index(inplace=True)
        # changes_df.drop(columns=changes_df.columns[0], axis=1, inplace=True)

        index_change = []
        for idx, _ in changes_df.iterrows():
            if(idx < changes_df.shape[0]-1):
                index_change.append([changes_df.iloc[idx]["index"], changes_df.iloc[idx+1]["index"], changes_df.iloc[idx]["gesture"]])

        gesture_range_df  = pd.DataFrame(index_change)
        gesture_range_df.columns = ["start","end","gesture"]

        label_mappings = {i: target_names[i] for i in range(len(target_names))}
        gesture_range_df["gesture"] = gesture_range_df["gesture"].map(label_mappings)

        return gesture_range_df
    
    pred_gesture_ranges_df = _convert_label_to_range(pred)
    gt_gesture_ranges_df = _convert_label_to_range(gt)

    # pred = alt.Chart(pred_gesture_ranges_df).mark_bar(clip=True).encode(
    #     x=alt.X('start', scale=alt.Scale(domain=[0,3000])),
    #     x2='end',
    #     y=alt.Y('sum(gesture)',title = "Gesture", axis=alt.Axis(labels=False)),
    #     color=alt.Color('gesture', scale=alt.Scale(scheme='dark2'))
    # ).properties(
    #     width=800,
    #     height=25,
    #     title="Prediction"
    # )

    # gt = alt.Chart(gt_gesture_ranges_df).mark_bar(clip=True).encode(
    #     x=alt.X('start', scale=alt.Scale(domain=[0,3000])),
    #     x2='end',
    #     y=alt.Y('sum(gesture)',title = "Gesture", axis=alt.Axis(labels=False)),
    #     color=alt.Color('gesture', scale=alt.Scale(scheme='dark2'))
    # ).properties(
    #     width=800,
    #     height=25,
    #     title="Ground Truth"
    # )

    # alt.vconcat(
    # gt.mark_bar(clip=True),
    # pred.mark_bar(clip=True),
    # )

    plt.scatter(np.arange(pred.shape[0]), pred, c='red')
    plt.scatter(np.arange(pred.shape[0]), gt, c='blue')
    plt.show()

def get_dataloaders(task: str,
                    subject_id_to_exclude: str,
                    observation_window: int,
                    prediction_window: int,
                    batch_size: int,
                    one_hot: bool,
                    class_names: List[str]):
    
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
    train_dataset = LOUO_Dataset(train_files_path, observation_window, prediction_window, onehot=one_hot, class_names=class_names)
    valid_dataset = LOUO_Dataset(valid_files_path, observation_window, prediction_window, onehot=one_hot, class_names=class_names)

    target_type = torch.float32 if one_hot else torch.long
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type))
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type)) 

    return train_dataloader, valid_dataloader  

def get_all_jigsaws_data():
    pass