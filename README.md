# CIFAR-10-Modeling-Deep-learning-Python
# CNN Training for CIFAR-10 Image Classification

This project trains a **Convolutional Neural Network (CNN)** for image classification using the CIFAR-10 dataset.  

## 📌 Steps

1. **Load CIFAR-10 Dataset**  
   - The dataset is divided into:
     - **Training set**: 50,000 images  
     - **Test set**: 10,000 images  

2. **Build the CNN Model**  
   - Use multiple **Convolutional Layers**, **Pooling Layers**, **Batch Normalization**, and **Dropout** to enhance model performance.  

3. **Apply Data Augmentation**  
   - Use **Data Augmentation** to increase the diversity of training data and improve generalization.  

4. **Optimize Training with Callbacks**  
   - Use **ModelCheckpoint** to save the best model.  
   - Use **EarlyStopping** to prevent overfitting.  

5. **Visualize Training Progress**  
   - Plot the **loss curve** and **accuracy curve** to monitor model learning.  

6. **Evaluate the Best Model**  
   - Load the best model and evaluate it on the **test dataset** for final performance assessment.

## Required Python Libraries

Before running the model, make sure to install and import the necessary libraries:

``` ruby
# Import essential libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix     
``` 
Library Overview:
- NumPy, Pandas: Used for data manipulation and numerical operations.

- TensorFlow / Keras: Provides tools to build and train deep learning models.

- Matplotlib, Seaborn: Used for visualizing data and model performance.

- Scikit-learn: Utilized for computing the confusion matrix to evaluate model performance.
## 📥 Load CIFAR-10 Dataset

To load the **CIFAR-10** dataset, use the following code:

```python
# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  
```
- CIFAR-10 is a commonly used image classification dataset that contains 10 different categories, such as airplanes, cars, cats, dogs, and others.
