
# 🌱 **Plant Disease Image Classification using SVM, CNN & ResNet50 (Deep Learning & Computer Vision)**

This project implements and compares three different approaches for classifying plant leaf diseases using the **PlantVillage dataset**:

1. **SVM with HOG (Hand-Engineered Features)**
2. **Custom Convolutional Neural Network (CNN)**
3. **Transfer Learning using ResNet50**

The goal is to evaluate classical machine learning methods vs. deep learning architectures in terms of **accuracy**, **scalability**, **computational efficiency**, and **real-world applicability**.

---

# 📁 **Dataset**

**PlantVillage Dataset**
Kaggle Link: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

* Contains **54,000+** high-quality leaf images
* Covers **38 different plant disease classes**
* Categories include:

  * Apple, Tomato, Potato, Corn, Grape, etc.
  * Diseases like Late Blight, Leaf Mold, Septoria, Early Blight, etc.

Dataset includes both **healthy** and **diseased leaves**, making it ideal for supervised image classification tasks.

---

# 🧹 **Data Preprocessing**

✔ Images resized to **64 × 64 pixels**
✔ Converted to **RGB** (for CNN/ResNet50) and **grayscale** (for HOG)
✔ Label encoding using **LabelEncoder()**
✔ Dataset split:

* **Train**
* **Validation**
* **Test**

⚖ **Class imbalance handled** using:

* Stratified splitting
* Random undersampling in classical SVM experiments

---

# 🧪 **Methods & Model Architectures**

## 1️⃣ **SVM + HOG Features**

A classical Computer Vision–based pipeline:

### 🔹 **HOG Feature Extraction**

* Converts image to grayscale
* Computes gradient intensities
* Extracts orientation histograms from local patches
* Produces a flattened feature vector (~1,764 features for 64×64 images)

### 🔹 **SVM Model**

* SVM Kernel: **Linear**
* Training subset: **1500 balanced samples**
* Motivation:

  * Fast training
  * Works well on small datasets
  * Good for baseline comparison

### 🟢 **Accuracy: ~80% on controlled subset**

✔ Advantages

* Lightweight & fast
* Good performance with small datasets

❌ Limitations

* Does not scale to large image datasets
* Cannot learn complex patterns
* Dependent on feature engineering

---

## 2️⃣ **Custom CNN (Convolutional Neural Network)**

A lightweight CNN built from scratch to train fully on the dataset.

### 🔨 **Architecture Details**

| Layer                        | Parameters              |
| ---------------------------- | ----------------------- |
| **Conv2D** (32 filters, 3×3) | ReLU + BatchNorm        |
| **Conv2D** (32 filters, 3×3) | ReLU                    |
| **MaxPool2D** (2×2)          | —                       |
| **Dropout (0.25)**           | —                       |
| **Conv2D** (64 filters, 3×3) | ReLU + BatchNorm        |
| **Conv2D** (64 filters, 3×3) | ReLU                    |
| **MaxPool2D**                | —                       |
| **Dropout (0.25)**           | —                       |
| **Flatten**                  | —                       |
| **Dense (128)**              | ReLU + Dropout(0.5)     |
| **Output Layer**             | Softmax (#classes = 38) |

### 🔧 **Training Setup**

* Optimizer: **Adam**
* Loss Function: **Categorical Crossentropy**
* Batch Size: **32**
* Epochs: **15–25**
* GPU-Accelerated Training

### 🟢 **Accuracy Achieved: ~84% (Test Set)**

✔ Learns feature hierarchies
✔ Good balance of speed & accuracy
✔ Performs significantly better than SVM

❌ Can overfit if augmentation is not applied
❌ Image size (64×64) limits representational power

---

## 3️⃣ **Transfer Learning – ResNet50**

Used as a feature extractor + small custom classification head.

### 🧱 **Architecture**

* Pretrained on **ImageNet**
* Base model: **ResNet50 (frozen)**
* Input size: **224 × 224** (resized from 64 × 64 → likely suboptimal)
* Custom Head:

  * GlobalAveragePooling2D
  * Dense(256, ReLU)
  * Dropout(0.5)
  * Dense(number_of_classes, Softmax)

### ⚠️ Observed Problems

❗ Using 64×64 images and upscaling results in major **low-resolution artifacts**
❗ Very few epochs, limited fine-tuning
❗ Large domain gap between plant leaves vs. ImageNet objects

### 🔴 **Accuracy: ~34%**

❗ Performs worst among all models
❗ Undertrained + insufficient fine-tuning

---

# 🔬 **Experiment Results Summary**

| Model                          | Test Accuracy | Notes                                                      |
| ------------------------------ | ------------- | ---------------------------------------------------------- |
| **SVM + HOG**                  | **80%**       | High accuracy for small fixed dataset, not scalable        |
| **Custom CNN**                 | **84%**       | Best performer overall; lightweight & robust               |
| **ResNet50 Transfer Learning** | **34%**       | Underperformer; needs proper fine-tuning and larger images |

---

# 🏁 **How to Run the Project**

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   ```

2. **Download the PlantVillage dataset**
   Place in:

   ```
   data/
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebooks / Scripts**

   * Preprocessing
   * CNN Training
   * ResNet50 Fine-tuning
   * Evaluation Scripts

---

# 📊 **Why ResNet50 Performed Poorly (Important Insight)**

✔ **Mismatched input size (64×64 → 224×224)**
✔ **Not enough training epochs**
✔ **No data augmentation**
✔ **Frozen layers prevent domain adaptation**
✔ **Plant leaves textures differ drastically from ImageNet objects**

---

# 📘 **Conclusion**

* **SVM + HOG** provides a strong classical baseline but fails on large-scale image learning tasks.
* **Custom CNN** gives the best performance (84%) due to its ability to learn hierarchical features directly from pixels.
* **Transfer Learning with ResNet50**, although powerful, performs poorly due to inadequate fine-tuning and low input resolution.

Final verdict:
👉 **Custom CNN** is optimal for this project given dataset size, preprocessing, and input resolution.

---

# 📚 **References**

1. PlantVillage Dataset, Hughes & Salathé, Kaggle, 2015
2. Kaiming He, Zhang, Ren & Sun – *Deep Residual Learning for Image Recognition*, CVPR 2016
3. Dalal & Triggs – *Histograms of Oriented Gradients*, CVPR 2005
4. François Chollet – *Deep Learning with Python*, Manning Publications


