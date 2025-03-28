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
