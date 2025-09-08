# Face Detection and Skin Type Classification Models

This repository contains a Jupyter Notebook demonstrating the implementation of two distinct deep learning models using transfer learning:
1.  A **Face Detection model** (binary classifier) built with TensorFlow and Keras.
2.  A **Skin Type Classification model** (multi-class classifier) built with PyTorch.

The notebook covers the entire pipeline for both models, from data preprocessing and augmentation to model training, fine-tuning, evaluation, and testing on custom images.

## Table of Contents
- [Model 1: Face Detection (TensorFlow/Keras)](#model-1-face-detection-tensorflowkeras)
  - [Objective](#objective)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training and Fine-Tuning](#training-and-fine-tuning)
  - [Usage](#usage)
- [Model 2: Skin Type Classification (PyTorch)](#model-2-skin-type-classification-pytorch)
  - [Objective](#objective-1)
  - [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
  - [Model Architecture](#model-architecture-1)
  - [Training and Evaluation](#training-and-evaluation)
  - [Usage](#usage-1)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)

---

## Model 1: Face Detection (TensorFlow/Keras)

### Objective
To build a binary image classifier that determines whether a given image contains a human face (`face`) or not (`non_face`). The classes are mapped as `{'face': 0, 'non_face': 1}`.

### Data Preprocessing
- **Initial Segregation**: A script iterates through an `original_dataset` directory. It inspects filenames for the keyword "human" to automatically segregate images into `dataset/face` and `dataset/non_face` subdirectories.
- **Data Augmentation**: Keras's `ImageDataGenerator` is used to preprocess the images and apply data augmentation to the training set. This helps the model generalize better. Augmentations include:
  - Rescaling pixel values to a `[0, 1]` range.
  - Random rotations, width/height shifts, shearing, and zooming.
  - Horizontal flipping.

### Model Architecture
The model leverages **transfer learning** with the **MobileNetV2** architecture, pretrained on the ImageNet dataset.
- The base MobileNetV2 model is used as a feature extractor with its top classification layer removed.
- The layers of the base model are initially frozen.
- A custom classifier head is added on top, consisting of:
  - `GlobalAveragePooling2D` layer to flatten the feature maps.
  - `Dropout` layers to prevent overfitting.
  - A `Dense` layer with `ReLU` activation.
  - A final `Dense` output layer with a `Sigmoid` activation for binary classification.

### Training and Fine-Tuning
The model is trained in two stages for optimal performance:
1.  **Initial Training**: Only the custom classifier head is trained for 10 epochs with the MobileNetV2 base frozen. The model quickly achieves high accuracy on the validation set.
2.  **Fine-Tuning**: The top 30 layers of the MobileNetV2 base are unfrozen. The model is then re-compiled with a much lower learning rate (`0.00001`) and trained for an additional 5 epochs to fine-tune the weights on the specific dataset.

The final trained model is saved as `face_detector.keras`.

### Usage
The notebook includes a testing section that demonstrates how to:
1.  Load the saved `.keras` model.
2.  Open and preprocess a single test image to match the model's input requirements (224x224, normalized).
3.  Use `model.predict()` to get a prediction and classify it as "Face Detected" or "Not a Face".

---

## Model 2: Skin Type Classification (PyTorch)

### Objective
To build a multi-class image classifier that identifies one of five skin types from a given image: **acne, burned, dry, normal, or oily**.

### Data Preprocessing and Augmentation
- **Data Loading**: PyTorch's `ImageFolder` and `DataLoader` are used to efficiently load the training, validation, and test datasets from the `dataset/Oily-Dry-Skin-Types` directory.
- **Data Augmentation**: `torchvision.transforms` is used to apply different sets of transformations for training and validation/testing:
  - **Training**: Includes `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, and `RandomRotation`.
  - **Validation/Test**: Includes `Resize` and `CenterCrop` for consistent evaluation.
  - Both sets include normalization with ImageNet's standard mean and standard deviation.

### Model Architecture
This model also uses **transfer learning** with the **ResNet50** architecture, pretrained on ImageNet.
- The pretrained ResNet50 model is loaded.
- The final fully connected layer (`model.fc`) is replaced with a new `torch.nn.Linear` layer with 5 output units, corresponding to the number of skin type classes.

### Training and Evaluation
- **Loss Function**: To handle class imbalance in the dataset, class weights are calculated and passed to the `nn.CrossEntropyLoss` function.
- **Optimizer**: The `optim.Adam` optimizer is used.
- **Scheduler**: A `ReduceLROnPlateau` learning rate scheduler is implemented to reduce the learning rate when validation loss plateaus.
- **Training Loop**: The model is trained for 30 epochs, with performance on the validation set checked at the end of each epoch.
- **Evaluation**: The model's performance is thoroughly evaluated using:
  - **Validation Accuracy**: Achieved **86%**.
  - **Test Accuracy**: Achieved **87.66%**.
  - **Classification Reports** and **Confusion Matrices** for both validation and test sets to provide detailed per-class metrics.

The final trained model's state dictionary is saved as `skin_type_classifier.pth`.

### Usage
The final section of the notebook demonstrates how to use the trained model for inference on custom images:
1.  Load the ResNet50 architecture and modify its final layer.
2.  Load the saved state dictionary (`.pth` file) into the model.
3.  Process custom test images using the defined test transforms.
4.  Predict the skin type and display the image with its predicted label.

---

## Technologies Used
- **TensorFlow & Keras**: For the Face Detection model.
- **PyTorch & Torchvision**: For the Skin Type Classification model.
- **Scikit-learn**: For generating evaluation metrics like classification reports and confusion matrices.
- **Matplotlib & PIL**: For image visualization and processing.
- **NumPy**: For numerical operations.
- **Jupyter Notebook**: For interactive development and presentation.

## How to Use
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install Dependencies**:
    It is recommended to create a virtual environment. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file containing libraries like `tensorflow`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`, `Pillow`.)*

3.  **Set Up Datasets**:
    - For the Face Detection model, create a folder named `original_dataset` and populate it with images. The preprocessing script will create the `dataset` folder.
    - For the Skin Type model, ensure your data is structured inside the `dataset/Oily-Dry-Skin-Types/` directory, with `train`, `valid`, and `test` subdirectories.

4.  **Run the Notebook**:
    Launch the Jupyter Notebook and run the cells sequentially to preprocess data, train, and test the models.
    ```bash
    jupyter notebook sample.ipynb
    ```
