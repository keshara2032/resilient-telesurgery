import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the input folder containing video frames and output folder for CSV files
input_folder = "./segmentation_masks/outputs"
output_folder_pca = "./segmentation_masks/pca_features"

# # Define the desired PCA dimensionality
# pca_components = 128  # Adjust this as needed

# # Function to load, resize, and flatten an image
# def load_resize_and_flatten_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is not None:
#         resized_image = cv2.resize(image, (320, 240))
#         return resized_image.flatten()
#     return None

# # Initialize an empty list to store flattened images
# flattened_images = []

# # Iterate through each folder in the input directory
# for folder_name in tqdm(os.listdir(input_folder), desc='reading images'):
#     folder_path = os.path.join(input_folder, folder_name, 'merged')
    
#     # Check if the item in the input directory is a folder
#     if os.path.isdir(folder_path):
#         # Iterate through each image in the folder
#         for image_name in sorted(os.listdir(folder_path)):
#             if image_name.endswith(".png"):
#                 image_path = os.path.join(folder_path, image_name)
#                 flattened_image = load_resize_and_flatten_image(image_path)
#                 if flattened_image is not None:
#                     flattened_images.append(flattened_image)

# # Convert the list of flattened images to a NumPy array
# image_matrix = np.array(flattened_images)

# # Perform PCA on the image matrix
# print
# pca = PCA(n_components=pca_components)
# pca.fit(image_matrix)

# # Iterate through each folder again to transform and save images
# for folder_name in os.listdir(input_folder):
#     folder_path = os.path.join(input_folder, folder_name, 'merged')
    
#     if os.path.isdir(folder_path):
#         # Create a CSV file with the same name as the folder
#         csv_filename = os.path.join(output_folder_pca, folder_name + ".csv")
        
#         with open(csv_filename, 'w', newline='') as csvfile:
#             csv_writer = csv.writer(csvfile)
            
#             # Iterate through each image in the folder
#             for image_name in sorted(os.listdir(folder_path)):
#                 if image_name.endswith(".png"):
#                     image_path = os.path.join(folder_path, image_name)
#                     flattened_image = load_resize_and_flatten_image(image_path)
#                     if flattened_image is not None:
#                         # Transform the image using PCA and save the result as a row in the CSV file
#                         transformed_image = pca.transform([flattened_image])
#                         csv_writer.writerow(transformed_image[0])

#         print(f"CSV file '{csv_filename}' created.")




# Define the input folder containing the original CSV files and the output folder for normalized CSV files
input_folder = output_folder_pca
output_folder = "./segmentation_masks/pca_features_normalized"

# Initialize an empty list to store data from all CSV files
all_data = []

# Step 1: Load all data from CSV files and concatenate them into a single array
for csv_filename in os.listdir(input_folder):
    if csv_filename.endswith(".csv"):
        csv_path = os.path.join(input_folder, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = [list(map(float, row)) for row in csv_reader]
            all_data.extend(data)

# Convert all_data to a NumPy array for fitting the scaler
all_data = np.array(all_data)

# Initialize a StandardScaler to perform Z-score normalization and fit it on all data
scaler = StandardScaler()
scaler.fit(all_data)

# Step 2: Iterate through each CSV file, transform data, and save to new CSV files
for csv_filename in os.listdir(input_folder):
    if csv_filename.endswith(".csv"):
        csv_path = os.path.join(input_folder, csv_filename)
        output_csv_path = os.path.join(output_folder, csv_filename)
        
        with open(csv_path, 'r') as csvfile, open(output_csv_path, 'w', newline='') as output_csvfile:
            csv_reader = csv.reader(csvfile)
            csv_writer = csv.writer(output_csvfile)
            
            data = [list(map(float, row)) for row in csv_reader]
            
            # Transform the data using the previously fitted StandardScaler
            normalized_data = scaler.transform(data)
            
            # Write the normalized data to the output CSV file
            csv_writer.writerows(normalized_data)
        
        print(f"Normalized CSV file '{output_csv_path}' created.")

print("Normalization completed.")








print("Processing completed.")
