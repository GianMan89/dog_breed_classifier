# dog_breed_classifier
Udacity Data Scientist Nanodegree: Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

# Disaster Response Pipeline Project

## Table of Contents
1. [Motivation](#motivation)
2. [Files in the Repository](#files)
3. [Instructions](#instructions)
	1. [Dependencies](#dependencies)
	2. [Executing Web Application](#execution)
4. [Project Definition](#definition)
5. [Analysis](#analysis)
6. [Conclusion](#conclusion)
7. [Licensing, Authors, and Acknowledgements](#licensing)

<a name="motivation"></a>
## 1. Motivation:

This project is the capstone project of the Data Science Nanodegree Program by Udacity.
This project uses Convolutional Neural Networks (CNNs)! The aim of this project was to build a pipeline to process real-world, user-supplied images. Given an image of a dog, the implemented algorithm identifies an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

The project includes a web app where the user can input a new image and get classification results.

<a name="files"></a>
## 2. Files in the Repository:

The repository consists of the following folders and files:

- ***faster_rcnn_inception_v2_coco_2018_01_28***: pretrained model files for the Faster RCN Inception V2 COCO model.
- ***haarscades**: pretrained open CV model for human face detection.
- ***images***: collection of images to test the app.
- ***saved_models***: saved model weights for the trained models.
- ***web_app***: contains HTML and Flask files that build and run the web-based API that classifies dog breeds.

The repository is structured as follows:

    faster_rcnn_inception_v2_coco_2018_01_28
    |- saved_model
    | |- saved_model.pb
    | |- variables
    |- checkpoint
    |- frozen_inference_graph.pb # pretrained model files for the Faster RCN Inception V2 COCO model
    |- model.ckpt.data-00000-of-00001
    |- model.ckpt.index
    |- model.ckpt.meta
    |- pipeline.config
    haarcascades
    |- haarcascade_frontalface_alt.xml # pretrained open CV model for human face detection
    images # collection of images to test the app
    |- American_water_spaniel_00648.jpg
    |- Brittany_02625.jpg
    |- Curly-coated_retriever_03896.jpg
    |- golden_retriever_1.jpg
    |- golden_retriever_2.jpg
    |- golden_retriever_3.jpg
    |- golden_retriever_4.jpg
    |- golden_retriever_5.jpg
    |- golden_retriever_6.jpg
    |- Inception-V3-architecture.png
    |- Labrador_retriever_06449.jpg
    |- Labrador_retriever_06455.jpg
    |- Labrador_retriever_06457.jpg
    |- sample_cnn.png
    |- sample_dog_output.png
    |- sample_human_1.jpg
    |- sample_human_2.jpg
    |- sample_human_3.jpg
    |- sample_human_output.png
    |- Welsh_springer_spaniel_08203.jpg
    saved_models # saved model weights for the trained models
    |- weights.best.from_scratch.hdf5 # saved model weights for the CNN implemented from scratch
    |- weights.best.inceptionv3.hdf5 # saved model weights for the pretrained Inception V3
    |- weights.best.VGG16.hdf5 # saved model weights for the pretrained VGG16
    web_app
    | - static
    | |- uploads # folder that is used to save the uploaded user images
    | - template
    | |- upload_image.html # main page of web app
    |- run.py # Flask file that runs app
    .gitignore
    dog_app.html # html-file with the analysis conducted for this project.
    dog_app.ipynb # Jupyter notebook file with the analysis conducted for this project.
    extract_bottleneck_features.py # script provided by Udacity to extract bottleneck features for the models
    LICENSE # MIT license for this repository
    poetry.lock # log file of all installed Python packages
    pyproject.toml # package summary and dependencies
    README.md

<a name="instructions"></a>
## 3. Instructions:
<a name="dependencies"></a>
### 1. Dependencies:

All relevant modules, versions and dependencies can be found in the 
poetry.lock and pyproject.toml

<a name="execution"></a>
### 2. Executing Web Application:

    1. Go to repository workspace directory

    2. Run your web app: `python run.py`

    3. Open http://127.0.0.1:3000/ in your browser.

    4. Click on `Choose File` and upload a jpg image.

    5. Click on `Submit` and enjoy the results.

<a name="definition"></a>
## 4. Project Definition:

The project consists of the following steps:

* Step 0: Import Datasets
* Step 1: Detect Humans
* Step 2: Detect Dogs
* Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
* Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
* Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
* Step 6: Write your Algorithm
* Step 7: Test Your Algorithm

This project implements a twofolded classification:
    1. decision if a dog or human is detected in the given image.
    2. if dog or human is detected, predict the most probable dog breed for this image.

A detailed description of each step can be found in dog_app.html and dog_app.ipynb.

<a name="analysis"></a>
## 5. Analysis:

A detailed description of the conducted analysis can be found in dog_app.html and dog_app.ipynb.

There are 8,351 dog photographs in total, which have been divided into 6,680 images for the training set, 835 images for the validation set, and 836 images for the testing set. The 133 dog breeds in the training set are not evenly distributed. Some dog breeds contain more than 70 photos, and others have fewer than 30. Due to the lesser number of training samples available for certain of these dog breeds, this may affect model performance.

Additionally, the inter-class and intra-class variations for dog breeds are smaller than compared to other animals or objects. Thus, we have to find some inconspicuous characteristics for the to be classified dog breeds. Since certain dog breeds can be quite similar to one another, the model may have trouble telling them apart. For instance, it may be challenging to distinguish between a "Brittany" and a "Welsh Springer Spaniel" due to their similar appearances.

<a name="conclusion"></a>
## 6. Conclusion:

For the first classification task, i.e., deciding if a human, a dog, or none of the former two is present in a given image, I first used OpenCV's implementation of Haar feature-based cascade classifiers, which showed some limitations when it comes to false positives, i.e., dogs that are misclassified as humans. An improvement I implemented was the pretrained Faster RCN Inception V2 COCO model, which performed better on the given test set, but still showed some limitations in cases where parts of a human body were present in a true dog image. To solve this, I implemented a pretrained ResNet-50 model to divide this classification in two steps, where the pretrained ResNet-50 model returns a binary classification whether a dog was detected and the pretrained Faster RCN Inception V2 COCO model does the same for the detection of humans.

For the second classification task, i.e., prediction of the most likely dog breed, I first implemented a convolutional neural network (CNN) from scratch. This CNN showed an accuracy of 4.55%. Using transfer learning and a pretrained VGG-16 model, I was able to reach an accuracy of 73.45% for this task. I was able to further improve this performance by using a pretrained Inception V3 model with 79.66% accuracy, which performs well above the goal of 60%.

However, with accuracies of 0%, it has some limitations in classifying certain dog breeds. Future work should put more emphasis on these performance imabalances.

Possible improvements to my model:

• Use data augmentation to make the breed detection algorithm more robust to different sizes, locations and distortions.
• Use more epochs to train a better prediction model.
• Use more training images to learn characteristic features that might not be included in the limited dataset.

<a name="licensing"></a>
## 7. Licensing, Authors, and Acknowledgements:

Code is under MIT license (see file).

Author: Gianluca Manca

Acknowledgements to:
* [Udacity](https://www.udacity.com/) for the Data Science Nanodegree program and image data.
