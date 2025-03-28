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
CIFAR-10 is a commonly used image classification dataset that contains 10 different categories, such as airplanes, cars, cats, dogs, and others.
## Visualizing CIFAR-10 Dataset
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Assuming X_train and y_train are predefined
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(5, 10)  # Create a 5x10 grid
for i in range(10):
    index = np.where(y_train == i)[0][:5]  # Get 5 images from each category
    for j in range(5):
        ax = plt.subplot(gs[j, i])
        ax.imshow(X_train[index[j], :, :], 'gray')
        ax.set_title(y_train[index[j]])
        ax.set_xticks([])
        ax.set_yticks([])
```
This code uses Matplotlib to display sample images from the CIFAR-10 training dataset. It selects 5 images from each of the 10 categories, using `np.where(y_train == i)[0][:5] `to fetch 5 images for each class and displays them in a 5x10 grid.
# Check the Shape of the Data
```python
print(X_train.shape)
```
Check the shape of X_train (the training dataset), which is typically (50000, 32, 32, 3), meaning there are 50,000 images, each with a size of 32×32 pixels and 3 color channels (RGB).
# Build a CNN (Convolutional Neural Network) Model
```python
model = Sequential()

model.add(BatchNormalization(input_shape=(32, 32, 3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
![image](https://github.com/user-attachments/assets/5aaf4891-6352-4008-8a26-27183c76dcc3)

```
This is a Convolutional Neural Network (CNN) with the following main architecture:

1. **BatchNormalization**: Normalizes the input images to help stabilize training.
2. **Conv2D (Convolutional Layer)**: Uses 32 filters of size 3×3 to extract features.
3. **BatchNormalization**: Makes the data distribution more stable and improves training effectiveness.
4. **Activation('relu')**: Applies the ReLU activation function, enabling the model to learn nonlinear features.
5. **MaxPool2D (Pooling Layer)**: Reduces dimensionality and computational load.
6. **Flatten**: Flattens the CNN output into a one-dimensional vector to pass into the fully connected layer.
7. **Dense (Fully Connected Layer)**:
   - 1024 neurons + ReLU: Learns higher-level features.
   - Dropout(0.5): Prevents overfitting.
   - Output Layer (10 classes + Softmax): Classifies the CIFAR-10 dataset into 10 categories.
# Compile the Model
```python
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['sparse_categorical_accuracy'])
![image](https://github.com/user-attachments/assets/41343373-9c00-4b0a-ad59-c2f44a57a5ce)
```
- **Loss Function**: `sparse_categorical_crossentropy` (suitable for labels that are not one-hot encoded).  
- **Optimizer**: `Adam` with a learning rate of 0.005.  
- **Evaluation Metric**: `sparse_categorical_accuracy` to measure prediction accuracy.
# Data Augmentation
```python
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             horizontal_flip=False,
                             vertical_flip=False
                             )
datagen.fit(X_train)![image](https://github.com/user-attachments/assets/ea416b75-801e-4902-8f97-ee64ee940d57)
```
**Data Augmentation**: Expands the dataset by applying random transformations such as rotation and translation to improve the model's generalization ability.  

- `rotation_range=10`: Randomly rotates images by ±10°.  
- `width_shift_range=0.3` & `height_shift_range=0.3`: Randomly shifts images horizontally and vertically by 30%.  
- `horizontal_flip=False` & `vertical_flip=False`: No horizontal or vertical flipping.    
