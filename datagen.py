import numpy as np
import pandas as pd
import os
import sys

all_tasks = ["Peg_Transfer", "Suturing", "Knot_Tying", "Needle_Passing"]
JIGSAWS_tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
class_names = {
    "Peg_Transfer": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    "Suturing": ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'],
    "Knot_Tyring": ['G1', 'G11', 'G12', 'G13', 'G14', 'G15'],
    "Needle_Passing": ["G1", 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
}
all_class_names = ["G1", 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15']

kinematic_feature_names = [ "PSML_position_x", "PSML_position_y", "PSML_position_z", \
            "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z", \
            "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w", \
            "PSML_gripper_angle", \
            "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
            "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z", \
            "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w", \
            "PSMR_gripper_angle"]
state_variables = ['left_holding', 'left_contact', 'right_holding', 'right_contact', 'needle_state']
state_variables_repeating_factor = 10

image_features_save_path = './image_features'

def generate_data(task: str):
    processed_data_path = "./ProcessedDatasets"
    root_path = os.path.join("./Datasets", "dV")
    task_path = os.path.join(root_path, task)
    task_path_target = os.path.join(processed_data_path, task)
    gestures_path = os.path.join(task_path, "gestures")
    video_path = os.path.join(task_path, "video")
    kinematics_path = os.path.join(task_path, "kinematics")
    transcriptions_path = os.path.join(task_path, "transcriptions")

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    if not os.path.exists(task_path_target):
        os.makedirs(task_path_target)

    videos = []    
    for file in os.listdir(gestures_path):
        if file.startswith(task):

            # read the gesture labels
            file_path = os.path.join(gestures_path, file)
            labels = pd.read_csv(file_path, sep=' ', index_col=None, header=None)
            if len(labels.columns) == 5:
                column_names = ['Start', "Stop", "MP", "Success", "X"]
                labels = labels.set_axis(column_names, axis=1).drop("X", axis=1)
            else:
                column_names = ['Start', "Stop", "MP", "Success"]
                labels = labels.set_axis(column_names, axis=1) 

            # read kinematic variables
            kinematics = pd.read_csv(os.path.join(kinematics_path, file[:-3] + 'csv'), index_col=None)

            # read state variables
            states_df = pd.read_csv(os.path.join(transcriptions_path, file), sep=' ', index_col=0, header=None)
            if len(states_df.columns) == 6:
                column_names = [*state_variables, "X"]
                states_df = states_df.set_axis(column_names, axis=1).drop("X", axis=1)
            else:
                column_names = state_variables
                states_df = states_df.set_axis(column_names, axis=1)

            # append states to the kinematics
            kinematics = pd.concat([kinematics, states_df], axis=1)
            kinematics = kinematics.ffill(axis = 0)

            # append labels to the kinematics
            kinematics['label'] = ['-'] * len(kinematics)
            for i, row in labels.iterrows():
                start, stop, mp = int(row['Start']), int(row['Stop']), row['MP']
                kinematics.loc[start:stop, 'label'] = mp

            # save compound file contaning kinematics, state variables and gesture label
            kinematics.to_csv(os.path.join(task_path_target, file[:-3] + 'csv'), index=False)

            # collect the video path files
            video_features_file_path = os.path.join(image_features_save_path, task, file[:-4] + '_Right' + '.npy')
            if not os.path.exists(video_features_file_path):
                video_features_file_path = os.path.join(image_features_save_path, task, file[:-4] + '_Left' + '.npy')
            if not os.path.exists(video_features_file_path):
                raise ValueError(f"The features for video file {os.path.basename(video_features_file_path)} does not exist")
            videos.append(video_features_file_path)
    
    with open(os.path.join(task_path_target, 'video_feature_files.txt'), 'w') as fp:
        for v in videos:
            fp.write(v + '\n')


if __name__ == "__main__":
    task = sys.argv[1]
    assert(task in all_tasks)
    generate_data(task)


