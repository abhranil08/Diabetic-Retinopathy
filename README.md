# Diabetic-Retinopathy
Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). It can cause blindness if left undiagnosed and untreated.

The condition affected over 90 million patients. 

The need for a comprehensive and automated method of diabetic retinopathy screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. 


# Data
Data is taken from the Kaggle Dataset. Unlike other kaggle datasets, Dataset is imbalanced and not preprocessed properly. Data needs to be preprocessed properly. The variance between images are very high. The images have to be rotated,cropped and resized properly in order to remove maximum noise. 
There are 5 class labels: No DR, Mild DR, Moderate DR, Severe DR, Proliferative DR.

# Architecture
Transfer Learning has been used, model Resnet50 and VGG16 implemented. The training accuracy coming is good according to previous results, around 75%. But the validation accuracy is not high as compared to the training accuracy (around 60%). The model is overfiting. So, feature extraction, parameter and more data will give better results. More CNN models will be tested in order to get maximum results.
