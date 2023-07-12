import numpy as np
import pandas as pd
import os
import sys

all_tasks = ["Peg_Transfer", "Suturing", "Knot_Tying", "Needle_Passing", "Pea_on_a_Peg", "Post_and_Sleeve"]

def generate_data(task: str):
    processed_data_path = "./ProcessedDatasets"
    root_path = os.path.join("./Datasets", "dV")
    task_path = os.path.join(root_path, task)
    task_path_target = os.path.join(processed_data_path, task)
    gestures_path = os.path.join(task_path, "gestures")
    video_path = os.path.join(task_path, "video")
    kinematics_path = os.path.join(task_path, "kinematics")

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    if not os.path.exists(task_path_target):
        os.makedirs(task_path_target)

    videos = []    
    for file in os.listdir(gestures_path):
        if file.startswith(task):
            file_path = os.path.join(gestures_path, file)
            labels = pd.read_csv(file_path, sep=' ', index_col=None, header=None)
            if len(labels.columns) == 5:
                column_names = ['Start', "Stop", "MP", "Success", "X"]
                labels = labels.set_axis(column_names, axis=1).drop("X", axis=1)
            else:
                column_names = ['Start', "Stop", "MP", "Success"]
                labels = labels.set_axis(column_names, axis=1)
            
            kinematics = pd.read_csv(os.path.join(kinematics_path, file[:-3] + 'csv'), index_col=None)
            kinematics['label'] = ['-'] * len(kinematics)
            for i, row in labels.iterrows():
                start, stop, mp = int(row['Start']), int(row['Stop']), row['MP']
                kinematics.loc[start:stop, 'label'] = mp
            kinematics = kinematics[kinematics['label'] != '-']
            kinematics.to_csv(os.path.join(task_path_target, file[:-3] + 'csv'), index=False)
            video_file_path = os.path.join(video_path, file[:-4] + '_Right' + '.avi')
            if not os.path.isfile(video_file_path):
                video_file_path = video_file_path.replace('avi', 'mp4')
                print(video_file_path)
            videos.append(video_file_path)
    with open(os.path.join(task_path_target, 'video_files.txt'), 'w') as fp:
        for v in videos:
            fp.write(v + '\n')


if __name__ == "__main__":
    task = sys.argv[1]
    assert(task in all_tasks)
    generate_data(task)


