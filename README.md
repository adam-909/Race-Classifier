# Race Classification using CNN

This project is a **deep learning model** that classifies human race based on facial images using a **Convolutional Neural Network (CNN)**. The dataset used is **FairFace**, and the model is built with **TensorFlow and Keras**. This is to aid in a bigger project promoting EDI

---

## Dataset

I used the **FairFace dataset**, which consists of labeled facial images with the following race categories:

- **Black**
- **East Asian**
- **Indian**
- **Latino/Hispanic**
- **Middle Eastern**
- **Southeast Asian**
- **White**

Images are preprocessed and normalised before being fed into the CNN model.

---

## Model Architecture

The CNN model consists of:

1. **Convolutional Layers** - Extract features from images.
2. **MaxPooling Layers** - Reduce dimensionality while preserving important information.
3. **Fully Connected (Dense) Layers** - Classify the images into race categories.
4. **Softmax Activation** - Outputs probabilities for each race class.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes
])
```

## To-Do List (Improvements for Better Accuracy & Speed)
1. Reduce Training Time
 - Enable GPU acceleration to speed up training.
 - Reduce batch size for systems with limited RAM.
 - Optimize steps_per_epoch to balance training speed.
2. Improve Model Accuracy
 - Implement data augmentation (rotation, flipping, brightness changes).
 - Use pretrained models (ResNet, VGG16) for better feature extraction.
 - Experiment with different architectures (more layers, better dropout).
 - Fine-tune learning rate for better convergence.
3. Fix Dataset Imbalance
 - Oversample underrepresented classes to balance race distribution.
 - Implement class weighting during training.
