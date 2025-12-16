# programming_language_miniproject
programming_language_miniproject CNN model(sports classification)

# Sport Images Classification Mini Project
# 21102049 Minhyuk Lee


# 1. Problem Definition & Dataset Descriptions

## 1.1 Background & Motivation

A huge amount of sports-related images is produced every day through news articles, sports media platforms, team social media accounts, and highlight videos.  
Most of these images need to be categorized by sport (e.g., football, basketball, tennis, etc.) in order to be managed, searched, or used for automated content recommendation.  

However, manually tagging thousands of images is time-consuming and inconsistent.  
To address this issue, an automated system that can classify the sport shown in an image would be extremely useful.

Convolutional Neural Networks (CNNs) are known to perform well in image classification tasks because they can learn visual patterns such as shapes, textures, objects, and environments.  
Different sports usually have distinguishable features such as courts, equipment, uniforms, and body poses.  
This makes the problem suitable for CNN-based classification.

---

## 1.2 Problem Definition

In this project, I aim to build a CNN model that automatically predicts the sport category of an input image.  
This is formulated as a **multi-class image classification** task.

- **Input**: a single RGB image containing a sports scene  
- **Output**: a sport label belonging to one of the categories in the dataset  
  (e.g., football, basketball, baseball, tennis, swimming, etc.)

Formally, the model learns a function:

\[
f(x) \rightarrow y
\]

where \(x\) is an image and \(y\) is one of the sport classes.

The main goals are:
1. To implement a **baseline CNN model** trained from scratch, and  
2. To build an **improved model** using transfer learning (e.g., ResNet18) and compare its performance against the baseline.

---

## 1.3 Why CNNs Are Suitable for This Task

Each sport contains repeated and recognizable visual patterns:

- **Environment differences**: football field, basketball court, swimming pool, tennis court  
- **Equipment**: balls, rackets, bats, nets, goals, etc.  
- **Uniforms**: different colors, designs, or protective gear  
- **Typical poses and actions**: shooting a basketball, kicking a football, swimming strokes, tennis serves, etc.

CNNs are effective at capturing such patterns because they learn hierarchical representations, from low-level shapes and textures to high-level semantic features.  
Therefore, CNNs are an appropriate choice for this type of classification problem.

---

## 1.4 Dataset Description

For this project, I use the **Sports Image Classification** dataset available on Kaggle.  
The dataset contains multiple folders, each representing a different sport category.  
Each folder includes real-world sports images collected from media sources.

- **Source**: Kaggle – Sports Image Classification Dataset  
- **Data type**: RGB sports images (JPEG/PNG format)  
- **Task**: multi-class classification  
- **Classes**: several sports categories (exact list will be confirmed after downloading in the next section)

After downloading the dataset, I will inspect the directory structure, check the number of images per class, and visualize a few samples.

To properly evaluate the model, the dataset will be split into:

- **Training set**: 70%  
- **Validation set**: 15%  
- **Test set**: 15%

This split ensures that the model has enough data for learning while also providing separate data for tuning and final evaluation.

## 2. Data Setup and Exploration
### 2.1 Kaggle API configuration
### 2.2 Downloading and extracting the sports dataset
### 2.3 Checking the folder structure in `data/`
### 2.4 Inspecting classes and a sample training image

## 3. Data Preprocessing and Dataloaders
### 3.1 Defining data transforms (train vs validation/test)
### 3.2 Creating PyTorch datasets with `ImageFolder`
### 3.3 Building dataloaders for training, validation, and test

## 4. Baseline CNN Model

Before experimenting with advanced pretrained architectures such as ResNet or VGG, I first implemented a simple custom convolutional neural network (CNN) to serve as the baseline model. This baseline network allows me to evaluate how a straightforward CNN performs on the sports image classification task and provides a reference point for later comparison with more sophisticated models.

### What kind of model is this?
This baseline model is **not** ResNet, VGG, EfficientNet, or any other predefined architecture.  
It is a **custom-built CNN** designed from scratch with the following goals:

- Keep the architecture simple and easy to interpret  
- Use standard convolution → batch normalization → ReLU → pooling blocks  
- Provide a reasonable baseline for comparison with pretrained models  
- Ensure that the model is lightweight enough to train quickly on Google Colab  

### Architecture overview
The model consists of:

- **Three convolutional blocks**
  - `Conv2D → BatchNorm → ReLU → MaxPool`
  - Feature map sizes reduce as: 224 → 112 → 56 → 28
- **A fully connected classifier**
  - `Flatten → Linear(128×28×28 → 256) → ReLU → Dropout(0.5)`
  - Final output layer with `num_classes` units

### Why use this model as a baseline?
- It is simple enough to train quickly and observe basic learning behavior.
- It provides a clear measurement of how much improvement is gained later by using pretrained architectures.
- It helps diagnose whether the dataset is easy or difficult before moving on to more complex models.
- It aligns with typical machine learning practice where a self-built baseline model is compared with transfer learning models.

This section defines the architecture and trains this baseline CNN. In Section 5, I compare its performance with an improved model using a pretrained network.

### 4.1 Baseline CNN architecture
### 4.2 Training setup (device, loss, optimizer, hyperparameters)
### 4.3 Training and validation loops for the baseline model
### 4.4 Training the baseline CNN model

## 5. Baseline Model Evaluation and Analysis
### 5.1 Learning curves of the baseline CNN
### 5.2 Final performance of the baseline model
### 5.3 Detailed classification metrics for the baseline model
### 5.4 Discussion of the baseline results

The baseline CNN clearly struggles with this task. Both the training and validation accuracy stay around 1%, which is close to random guessing for a 100-class classification problem. The loss curves also do not show a clear downward trend, indicating that the model is not learning useful representations from the data.

There are several reasons for this behavior:

- The dataset is **large-scale and fine-grained**: there are 100 different sports classes, many of which look visually similar (e.g., different types of racing or different martial arts).
- The baseline network is **very shallow and simple** compared to modern architectures such as ResNet or EfficientNet. It does not have enough capacity to capture complex visual patterns in the images.
- The model is trained **from scratch**, without any pretrained weights. With limited training epochs and limited data per class, it is difficult for such a small model to learn discriminative features.

As a result, the baseline model provides a useful **lower bound** on performance.  
In the next section, I will introduce an improved model based on a pretrained ResNet architecture and compare how much performance gain we can obtain by using transfer learning.

## 6. Improved Model with Pretrained ResNet18
### 6.1 Transfer learning strategy with ResNet18

To significantly improve over the simple baseline CNN, I use **transfer learning** with a pretrained ResNet18 model from torchvision.

ResNet18 is a much deeper architecture that uses *residual connections* to stabilize the training of very deep networks. It is originally trained on the ImageNet dataset (1.2M images, 1000 classes), so its convolutional layers already contain rich visual features.

In this project, I use the following transfer learning strategy:

- Load a **pretrained ResNet18** (weights trained on ImageNet).
- **Freeze all convolutional layers**, so their pretrained weights are not updated.
- **Replace the final fully connected layer** with a new `Linear` layer that outputs 100 sports classes from this dataset.
- Train only the new classification head using the sports dataset.

This approach allows the model to reuse powerful pretrained features while keeping the number of trainable parameters small, which makes training faster and more stable compared to training a deep network from scratch.

### 6.2 Building the ResNet18-based classifier and training setup
### 6.3 Training the improved ResNet18 model
### 6.4 Evaluation of the improved model on validation and test sets

## 7. Comparison between the baseline CNN and the ResNet18 model
### 7.2 Learning curves of the improved ResNet18 model
### 7.3 Final validation and test performance comparison
### 7.4 Confusion matrix of the improved ResNet18 model
### 7.5 Failure case examples of the ResNet18 model

To better understand the limitations of the improved model, I also visualized several
misclassified validation images (failure cases). For each example, the figure shows the
true sport label and the label predicted by the ResNet18 model.

These qualitative examples help explain where the model still struggles, even though the
overall validation and test accuracies are high.

## 8. Conclusion and future work

### 8.1 Summary of findings

In this mini project, I tackled a **100-class sports image classification** problem using a CNN-based approach.  
The dataset consists of more than 13k training images, 500 validation images, and 500 test images, and each class corresponds to a different sport (e.g., *basketball*, *parallel bar*, *harness racing*, …).

I compared two different models:

- a **baseline CNN** with three convolutional blocks and a small classifier, trained from scratch
- an **improved model** based on **ResNet18** with transfer learning from ImageNet

The main observations are:

- The **baseline CNN** stayed around **1% accuracy** on both the validation and test sets, which is essentially random guessing for 100 classes.  
  The learning curves showed almost no improvement over epochs, and the confusion matrix was almost uniform.  
  This indicates that the shallow CNN was not able to learn meaningful representations for such a challenging, fine-grained dataset.
- The **ResNet18 model** achieved around **78.4% validation accuracy** and **81% test accuracy**.  
  The loss continuously decreased and the accuracy steadily increased during training, and the normalized confusion matrix showed a strong diagonal structure, meaning that many classes were correctly predicted.

These results clearly demonstrate the importance of **model capacity** and **transfer learning** for real-world image classification tasks with many classes and large intra-class variation.  
While the baseline CNN serves as a simple reference, the ResNet18 model successfully leverages pretrained features from ImageNet and adapts them to the sports domain.

Data augmentation (random resized crop and horizontal flip) also played an important role.  
It helped the ResNet18 model generalize better by exposing it to slightly different views of the same sport scenes, which is reflected in the relatively small gap between training and validation accuracy.

### 8.2 Limitations and failure cases

Although the ResNet18 model performs well overall, there are still several limitations:

- Some **visually similar sports** (for example, sports that take place on similar fields or involve similar equipment) are still confused with each other.  
  This can be seen as off-diagonal blocks in the confusion matrix.
- The model was trained with a fixed input resolution (224×224).  
  Small details that distinguish certain sports (e.g., the exact type of apparatus or ball) may be lost during resizing and cropping.
- I only fine-tuned the final classification layer.  
  In some difficult classes, the pretrained features from ImageNet may not be fully optimal for sports-specific patterns.

### 8.3 Future work

There are several directions to further improve this work:

1. **Fine-tuning more layers of ResNet18**  
   Instead of freezing all convolutional layers, I could unfreeze some of the deeper blocks and fine-tune them on the sports dataset.  
   This would allow the model to adapt its high-level features more specifically to sports scenes.

2. **Exploring stronger data augmentation**  
   Techniques such as random rotation, color jitter, random erasing, or MixUp/CutMix could improve robustness to viewpoint changes, lighting conditions, and occlusions.

3. **Trying other architectures**  
   It would be interesting to compare ResNet18 with other modern architectures such as EfficientNet, DenseNet, or vision transformers (ViT) to see whether they can further boost performance.

4. **Class-balanced training or focal loss**  
   If some sports classes are underrepresented, using class-balanced sampling or losses that focus more on difficult examples could improve performance on minority classes.

Overall, this project gave me hands-on experience with **PyTorch, CNN training, transfer learning, and model evaluation** on a realistic multi-class image dataset.  
The large performance gap between the baseline CNN and the ResNet18 model highlights how important it is to choose an appropriate architecture and to make effective use of pretrained models in practical deep learning applications.

## 8.4 Failure Case Analysis

To better understand the limitations of the improved ResNet18 model, we examined several **misclassified validation samples**.  
Failure cases provide valuable insights into what types of images the model struggles with and why such mistakes occur.

###  Observations from misclassified examples

1. **Images with unusual angles or occlusion**  
   Several images were captured from non-standard viewpoints or had partially invisible key features, making correct classification difficult.

2. **Inter-class similarity**  
   Some sports (e.g., *parallel bar* vs *balance beam*, *archery* vs *axe throwing*) share visually similar poses or backgrounds, which led the model to confuse between related classes.

3. **Low-light or motion blur**  
   A few images had notable blur or lighting issues that degraded feature visibility, reducing the model’s confidence.

4. **Background dominance**  
   In some samples, the sport-specific objects or actions were small, and the background occupied most of the image.  
   The model likely focused on irrelevant features, causing incorrect predictions.

###  Why failure cases matter

- They help identify which classes require **more data augmentation**.
- They reveal that some classes may need **better-balanced training samples**.
- They highlight the potential benefit of **attention-based models** (e.g., ViT, EfficientNet).
- They guide future improvements such as:
  - Enhanced preprocessing
  - Class-specific augmentation
  - Better architecture or fine-tuning depth

###  Example misclassified images

Below are representative failure case examples:

- Each image shows:
  - **True label** (ground truth)
  - **Predicted label** (model output)
- Images illustrate typical patterns in the mistakes discussed above.

*(The corresponding plots appear below this markdown cell.)*
