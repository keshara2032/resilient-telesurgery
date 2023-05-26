
import cv2
import os
import numpy as np
import sys

def extract_frames(filename, extract_rules, save_directory, video_directory):

    cap = cv2.VideoCapture(video_directory + filename + '.avi')
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("total-frames: ",total_frames)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    else:


        for rule in extract_rules: # ['0 173 S1 True', '173 265 S2 True', '265 353 S3 True',...]
            
            rule = rule.split(' ')
            start_frame = int(rule[0]) # 0 - start frame
            end_frame = int(rule[1]) # 173 - end frame
            gesture = rule[2] # S1 - gesture
            print(start_frame,end_frame,gesture)

            frame_index = start_frame

            if not os.path.exists(save_directory+gesture+"/"+filename):
                os.makedirs(save_directory+gesture+"/"+filename)
                

            frame_save_path = save_directory+gesture+"/"+filename

            while(frame_index <= end_frame):

                cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_index)   
                ret, frame = cap.read()
                

                if(ret):
                    cv2.imwrite(frame_save_path + "/frame_"+str(frame_index)+"_"+gesture +'.jpg', frame)
                # Display the resulting frame
                # cv2.imshow('Frame: ' + str(frame_index) + gesture,frame)

                frame_index += 1
                # cv2.waitKey()
                # cv2.destroyAllWindows()

    cap.release()


# assign directory

gesture_labels_directory = './dataset/Peg_Transfer/gestures/'
video_directory = './dataset/Peg_Transfer/video/'
kinematic_directory = './dataset/Peg_Transfer/kinematics/'
video_save_directory = './new_dataset/video/'
kinemati_save_directory = './new_dataset/kinematic/'


if __name__ == "__main__":

    # iterate over files in
    # that directory
    for filename in os.listdir(gesture_labels_directory):
        f = os.path.join(gesture_labels_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            filename_label = filename.split('.')[0] # filename of trial without extension
            print(filename_label) # Peg_Transfer_S01_T01

            with open(f) as file: # open Peg_Transfer_S01_T01.txt
                lines = [line.rstrip() for line in file]
                gesture_label_rules = lines
                # print(gesture_label_rules) # content of Peg_Transfer_S01_T01.txt ==>['0 173 S1 True', '173 265 S2 True', '265 353 S3 True',...]

                extract_frames(filename_label,gesture_label_rules,video_save_directory,video_directory)



