# Mushroom Identifier


## Overview
This is a deep learning project aimed at classifying mushroom species using pre-trained Convolutional Neural Networks (CNNs). With a dataset of over 77,000 images spanning 100 species, this project achieves high accuracy of 83%, offering a reliable tool for identifying mushrooms.

## Features
- Utilizes DenseNet201, a pre-trained CNN, for high accuracy in multi-class classification.
- Processes a diverse dataset with over 100 mushroom species.
- Implements transfer learning and fine-tuning techniques.
- Achieves an accuracy of 83% in classifying mushroom species.

## Dataset
The dataset consists of 77,382 images of 100 mushroom species. The images are preprocessed and augmented for better generalization. Key dataset features:
- Average of 774 images per class.
- Images resized to 224x224 pixels.
- Data split: 80% training, 20% validation.
<img width="936" alt="image" src="https://github.com/user-attachments/assets/d4a3d599-149d-4449-a65c-9de1c65f4046" />
<img width="936" alt="image" src="https://github.com/user-attachments/assets/deacb0b6-6830-474f-990b-d045f352a704" />



Source: [Kaggle Mushroom Species Dataset](https://www.kaggle.com/datasets/thehir0/mushroom-species).

## Methodology
1. **Pre-trained Models:**
   - Evaluated MobileNet, MobileNetV3, DenseNet169, and DenseNet201.
   - DenseNet201 outperformed others with 83% accuracy.
2. **Architecture:**
   - Base: DenseNet201 with custom layers.
   - Features:
     - Global Average Pooling.
     - Dense Layer with ReLU activation.
     - Dropout for regularization.
     - Final softmax output layer.
3. **Training Strategy:**
   - Transfer learning: Fine-tuned DenseNet201 with last dense block unfreezed.
   - Loss reduction via `ReduceLROnPlateau` and `EarlyStopping` callbacks.
<img width="331" alt="image" src="https://github.com/user-attachments/assets/95f493ab-86c1-4921-a14f-8435f56a6ed6" />
<img width="459" alt="image" src="https://github.com/user-attachments/assets/ed829474-31b8-4073-abc2-9f298c06ee3c" />


## Results
- DenseNet201 achieved a training accuracy of 88.86% and a validation accuracy of 83.78%.
- Fine-tuned model metrics:
  - Weighted Precision: 83.82%
  - Weighted Recall: 83.78%
  - Weighted F1-Score: 83.38%
- Metrics per class
![examples](https://github.com/user-attachments/assets/a5876c89-91da-45e3-a823-a08bb0245a19)
- Confusion Matrix
![confusion matrix](https://github.com/user-attachments/assets/806f99d1-ec3d-4e1a-a4ea-8c90d2b73359)



## Tools & Technologies
- **Libraries:** TensorFlow, Keras
- **Environment:**
  - Local machine (MacBook Air with M2 chip)
  - Google Colab (A100 GPU)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mushroom-identifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mushroom-identifier
   ```
3. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib scikit-learn
   ```
4. Replace the file paths to your file paths and run the training notebook:
   ```bash
   jupyter notebook denseNet_201_train.ipynb
   ```
Or download the .ipynb files and upload to Google Colab.

## Future Work
- Further clean the dataset to reduce noise.
- Implement object detection for better localization of mushrooms in images.
- Explore models with more trainable parameters for enhanced performance.

## References
1. Jitdumrong Preechasuk et al., "Image Analysis of Mushroom Types Classification by CNNs," AICCC 2019.
2. N. Kiss and L. Czùni, "Mushroom Image Classification with CNNs," ISPA 2021.
3. Huang, Gao et al., "Densely Connected Convolutional Networks," CVPR 2017.

## License
This project is licensed under the MIT License.


