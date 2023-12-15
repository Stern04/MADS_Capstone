# Using Pytorch CNNs to Detect Pedestrians in Autonomous Vehicle Data
This repository focuses on data originally shared from the 2019 Kaggle Competition hosted by Lyft "3D Object Detection for Autonomous Vehicles" which can be found at https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/overview. This project showcases a Pytorch implementation of Image Classification and Object Detection.
![Alt text](Report_examples\neuscenes.PNG)
## The Goal:

The goal of this repository is related but different to the goal set by the original Kaggle Competition. This involved heavy use of lidar data in conjuction with imagery data. This project, instead, soley focuses on using the image data for imagery based work.

In particular, the goal of this repository is to take the few hundred thousand images provided by Lyft and use them as an introductory to applications of CNN models for Image Classification and Object Detection. I thought it would be interesting and realistic to use the image data from Lyft as it was collected in a real world scenario which means it comes with the nuances and challenges that a Data Scientist would face in real world applications. Specifically, I decided that I would take the initial approach of binary classification to make the project scope more manageable, and this resulted in a focus on identifying the Pedestrian category/class. 

While this goes beyond some of the most basic Image Classification exercises one might find with common datasets like MNIST or CIFAR, it is still designed as an entry point to using CNNs on image data as well as common image preprocessing techniques. In it's initial phase (Dec 2023), the project will focus more on the application, data manipulation and model inference results rather than focusing on theory or experimenting with and across CNN architectures.

#### The initial phase yielded many interesting patterns and required domain specific knowledge in order to transform the raw dataset into something meaningful. A few of these were captured in visualizations below:
- The pie chart on the left shows one of the top performing Resnet50 model's most common error based on object category
- The upper right image (credit to University College London https://github.com/UCL/MPHY0026/blob/master/doc/calibration/camera_calibration.rst) depicts some of the complex matrix transformations that are required to translate between World Space (default) and Image Space
- The lower right image shows the full image space of a given image in our dataset denoted by the x and y axis. The red shading in the middle is a heatmap of occurences of our positive category (Pedestrian) which highlights the most common locations in the imagery that this category occurs.

<img src="Report_examples\merged_pic.PNG" width="1000" height="500">

## Project main Features:
### lyft_preprocessing.ipynb
### lyft_training_inference.ipynb
### create_dir_tree.py

The project is currently centered around two Jupyter Notebooks: lyft_preprocessing and lyft_training_inference. As their name suggests, the former is devoted to the many, tedious image preprocessing steps required to get images ready for model ingestion. The latter walks the user through using a plug and play Pytorch CNN model to perform Image Classification and Object Detection.

Each Notebook has many, detailed markdown cells within it that will help guide the user through the process. Additionally, each custom function has been well documented. 

NOTE: The data itself does not come with the repository. The full dataset (both images and lidar) is very large. I have left more details about data access in the Data section.

NOTE: You will need to mimic a particular file structure in combination with a set user directory path that the Notebooks expect. More in the How to Use the Project section.

# 0. Credits | Contact Information | Data Access
### Author: 
Chris Sternberger (sternbec@umich.edu | sternberger.chris@gmail.com)
### Data source
This project uses data sourced from Lyft, which is publicly available via their 2019 Kaggle Competition at https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/data
### Usage Rights & Restrictions
This project and associated data is available under MIT License.

# 1. Download the Repository

```
git clone https://github.com/Stern04/MADS_Capstone.git
```

# 2. The Data


