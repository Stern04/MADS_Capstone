import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

# List of directories to create
directories = [
    "Notebooks",
    "Test/data",
    "Test/images",
    "Test/lidar",
    "Test/maps",
    "Train/data",
    "Train/images",
    "Train/images/images_original",
    "Train/images/images_original/img_bx_lookup",
    "Train/images/Image_Classification",
    "Train/images/Image_Classification/train_cropped_images",
    "Train/images/Image_Classification/train_cropped_images/224",
    "Train/images/Image_Classification/train_cropped_images/224/animal_224",
    "Train/images/Image_Classification/train_cropped_images/224/bicycle_224",
    "Train/images/Image_Classification/train_cropped_images/224/bus_224",
    "Train/images/Image_Classification/train_cropped_images/224/emergency_vehicle_224",
    "Train/images/Image_Classification/train_cropped_images/224/motorcycle_224",
    "Train/images/Image_Classification/train_cropped_images/224/other_vehicle_224",
    "Train/images/Image_Classification/train_cropped_images/224/pedestrian_224",
    "Train/images/Image_Classification/train_cropped_images/224/truck_224",
    "Train/images/Image_Classification/train_cropped_images/animal",
    "Train/images/Image_Classification/train_cropped_images/bicycle",
    "Train/images/Image_Classification/train_cropped_images/bus",
    "Train/images/Image_Classification/train_cropped_images/emergency_vehicle",
    "Train/images/Image_Classification/train_cropped_images/motorcycle",
    "Train/images/Image_Classification/train_cropped_images/other_vehicle",
    "Train/images/Image_Classification/train_cropped_images/pedestrian",
    "Train/images/Image_Classification/train_cropped_images/truck",
    "Train/images/Image_Classification/validation_cropped_images",
    "Train/images/Image_Classification/validation_cropped_images/224",
    "Train/images/Image_Classification/validation_cropped_images/224/animal",
    "Train/images/Image_Classification/validation_cropped_images/224/bicycle",
    "Train/images/Image_Classification/validation_cropped_images/224/bus",
    "Train/images/Image_Classification/validation_cropped_images/224/emergency",
    "Train/images/Image_Classification/validation_cropped_images/224/motorcycle",
    "Train/images/Image_Classification/validation_cropped_images/224/other_vehicle",
    "Train/images/Image_Classification/validation_cropped_images/224/pedestrian",
    "Train/images/Image_Classification/validation_cropped_images/224/truck",
    "Train/images/Image_Classification/validation_cropped_images/animal_cropped",
    "Train/images/Image_Classification/validation_cropped_images/bicycle_cropped",
    "Train/images/Image_Classification/validation_cropped_images/bus_cropped",
    "Train/images/Image_Classification/validation_cropped_images/emergency_vehicle_cropped",
    "Train/images/Image_Classification/validation_cropped_images/motorcycle_cropped",
    "Train/images/Image_Classification/validation_cropped_images/other_vehicle_cropped",
    "Train/images/Image_Classification/validation_cropped_images/pedestrian_cropped",
    "Train/images/Image_Classification/validation_cropped_images/truck_cropped",
    "Train/images/Object_Detection",
    "Train/images/Object_Detection/img_bx_lookup",
    "Train/lidar",
    "Train/maps",
    "Train/models",
]

for dir in directories:
    create_directory(dir)

print("Directory structure created successfully.")
