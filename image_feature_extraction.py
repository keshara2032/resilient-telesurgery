import torch
from torch import optim, nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights







model = resnet50(weights=ResNet50_Weights.DEFAULT)

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