# Melanoma Skin Cancer Detection

## Introduction
- **Melanoma** is the least common skin cancer but responsible for 75% of skin cancer deaths.
- **Estimated new cases in 2022:** 99,780
- **Estimated deaths in 2022:** 7,650
- Detection in early stages helps in effective treatment and can save lives.

## Need for Melanoma Detection
- The objective of this project is to predict whether a patient has Melanoma, given a lesion image.
- A binary classification problem to classify a given image as Benign (non-cancerous) or Malignant (cancerous).

## Objective
- Referred from the official dataset of the SIIM-ISIC Melanoma Classification Challenge.
- **Total training images:** 33,126
- **Features:**
  1. `image_name` - unique identifier, points to filename of related image
  2. `patient_id` - unique patient identifier
  3. `sex` - the sex of the patient
  4. `age_approx` - approximate patient age at time of imaging
  5. `anatom_site_general_challenge` - location of imaged site
  6. `diagnosis` - detailed diagnosis information
  7. `benign_malignant` - indicator of malignancy of imaged lesion
  8. `target` - binarized version of the target variable

## Dataset
- **Training images:** 33,126

## Model
- **Compiling and training**
- **Cleaning & EDA**
- **Data Loading**

### Cleaning & EDA
1. **Removing columns that are not required for our model.**
2. **Model design flow.**
3. **Data cleaning steps:**
   - Removed images corresponding to unknown and less common categories in the `diagnosis` feature.

## Challenges of the Dataset
- **Highly imbalanced dataset:**
  - 32,524 Benign images (98.23%)
  - 584 Malignant images (1.763%)

### Our Solutions for Data Imbalance
- **Image Augmentation:**
  - Artificially creating new images from existing images by applying geometrical transformations (horizontal, vertical) and noise (gaussian blur, brightness, scaling).
- **Stratified Sampling:**
  - Ensuring the ratio between the target classes remains the same as in the full dataset during train-test split.

## Modeling Approaches
### Support Vector Classification (SVC)
- **Disadvantages:**
  - Not suitable for large datasets due to high training complexity.
  - Performs poorly in imbalanced datasets.

### Convolutional Neural Network (CNN)
- **Base Model:**
  - Simple CNN with 5 layers.
  - Activation Function: ReLU for hidden layers and Sigmoid for the last layer.
  - Achieved 92% accuracy.

### Transfer Learning and EfficientNetB4
- **Transfer Learning:**
  - Utilizes pre-trained model weights for new problems.
  - EfficientNet model by Google is used for image classification.

## Handling Overfitting
1. **Dropout Regularization:** Randomly removes neurons during training.
2. **Image Augmentation:** Creates new training images through various processing techniques.

## Training, Validation, and Results
| Metric            | Training | Validation |
|-------------------|----------|------------|
| Loss (Binary cross entropy) | 0.1471   | 0.1619     |
| Accuracy          | 0.9420   | 0.9349     |
| AUC               | 0.9545   | 0.9419     |

## Conclusion
- Detected Melanoma using image classification and CNNs with an accuracy & AUC-ROC of around 95%.
- Stratified sampling and data augmentation improved performance by nearly 2%.

## Future Considerations
- Image preprocessing for removing artifacts.
- Including other input features like Sex, Age, and location for classification.
- Implementing advanced CNN models like EfficientNetB7 on TPUs.

## Preventing Melanoma
- Wearing hats, goggles, long sleeve shirts (protective clothing), using sunscreen lotion, and regular doctor/self-checkups.

## References
1. [Melanoma Statistics](https://seer.cancer.gov/statfacts/html/melan.html)
2. [Skin Cancer Statistics](https://www.wcrf.org/cancer-trends/skin-cancer-statistics/)
3. [Research Paper](https://arxiv.org/abs/1412.6980)
4. [ISIC Challenge](https://challenge2020.isic-archive.com/)
5. [EfficientNet Research](https://arxiv.org/abs/1905.11946)
