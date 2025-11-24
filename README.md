# 🌱 **Plant Disease Image Classification: SVM, CNN &amp; ResNet50 (Deep Learning &amp; Computer Vision)**

This project tries out three ways to figure out plant leaf diseases using the **PlantVillage dataset**:

1.  **SVM with HOG (Using Basic Image Features)**
2.  **A Simple Neural Network (CNN)**
3.  **Using a Pre-Made Network (ResNet50)**

The main thing is to see if older methods compare to newer deep learning methods when it comes to how well they work (**accuracy**), how much they can handle (**scalability**), how fast they are (**computational efficiency**), and if they're useful in the real world.

---

# 📁 **Dataset**

**PlantVillage Dataset**
Kaggle Link: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

*   Over **54,000** good quality leaf pictures
*   **38** kinds of plant diseases
*   Includes:

    *   Apple, Tomato, Potato, Corn, Grape, etc.
    *   Diseases like Late Blight, Leaf Mold, Septoria, Early Blight, etc.

There are both **healthy** and **sick leaves**, which is good for teaching a computer to tell the difference.

---

# 🧹 **Getting the Data Ready**

✔ Pictures made smaller: **64 × 64 pixels**
✔ Made sure the pictures had color (**RGB** for CNN/ResNet50) and also made them black and white (for HOG)
✔ Gave each disease a number using **LabelEncoder()**
✔ Split the data into:

*   **Train** (to train the model)
*   **Validation** (to check the model is training well)
*   **Test** (to evaluate the model)

⚖ **Made sure each disease had about the same number of pictures** by:

*   Splitting carefully
*   Taking out some pictures of common diseases for the SVM

---

# 🧪 **How We Did It: Models**

## 1️⃣ **SVM + HOG Features**

This is an older, simple method:

### 🔹 **HOG: Figuring Out What's in the Picture**

*   Makes the image black and white
*   Figures out how the colors change in the image
*   Looks at small areas and figures out the main directions of the changes
*   Turns all this into a list of numbers (~1,764 numbers for each 64x64 picture)

### 🔹 **SVM Model**

*   SVM is set to use a **Linear** way of separating the data
*   Only used **1500** pictures that had an even number of each disease
*   Why?

    *   Trains fast
    *   Works okay with not many pictures
    *   Good to compare against other methods

### 🟢 **Accuracy: ~80% on a small dataset**

✔ Good things:

*   Quick and easy
*   Works well with a small set of pictures

❌ Bad things:

*   Doesn't work as great with many pictures
*   Can't learn very complicated things
*   Needs someone to figure out what features to look at

---

## 2️⃣ **Simple Neural Network (CNN)**

A basic CNN that I built ourselves lets it learn directly from the images.

### 🔨 **Network Layout**

| Layer                        | What it Does        |
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

### 🔧 **How I Taught It**

*   Used **Adam** to improve it little by little
*   Tried to make it get the right answer with **Categorical Crossentropy**
*   Looked at **32** pictures at a time
*   Did this **15–25** times
*   Used a fast computer chip (**GPU**)

### 🟢 **Accuracy: ~84% (Test Set)**

✔ Good at learning what to look for
✔ Good mix of speed and accuracy
✔ Better than the older method

❌ Can get too specific if you don't change the images up a bit
❌ Working with small images (64x64) limits what it can learn

---

## 3️⃣ **Using a Pre-Made Deep Learning Model – ResNet50**

Used a ready-made network to understand the pictures.

### 🧱 **How It's Set Up**

*   Learned basic image stuff from **ImageNet**
*   Used **ResNet50 (but didn't change it much)**
*   Pictures were **224 × 224** (had to make them bigger from 64 × 64 -&gt; which probably wasn't ideal)
*   Added a simple head:

    *   GlobalAveragePooling2D
    *   Dense(256, ReLU)
    *   Dropout(0.5)
    *   Dense(number_of_classes, Softmax)

### ⚠️ What Went Wrong

❗ Making the pictures bigger (from 64x64) made them look **blurry**
❗ Didn't train it enough
❗ Plant leaves are very different from what ImageNet knows

### 🔴 **Accuracy: ~34%**

❗ Worst of all the methods
❗ Didn't train it enough

---

# 🔬 **Results**

| Model                          | Test Accuracy | Notes                                                        |
| ------------------------------ | ------------- | ------------------------------------------------------------ |
| **SVM + HOG**                  | **80%**       | Good for a fixed set of pictures, but doesn't scale up well |
| **Simple CNN**                 | **84%**       | Best one overall; quick and accurate                         |
| **ResNet50** | **34%**       | Not so good; needs more training and better pictures   |

---

# 🏁 **How to Run This Yourself**

1.  **Get the code**

    ```bash
    git clone (https://github.com/kaivalya18/MLPROJECT/tree/main);
    ```

2.  **Get the PlantVillage pictures**
3.  **Install what you need**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the code**

    *   Preprocessing
    *   CNN Training
    *   Fine-tuning with ResNet50
    *   Scripts to see how well the models did

---

# 📊 **Why ResNet50 Didn't Work Well**

✔  **Pictures were too small (had to blow them up)**
✔  **Didn't train it long enough**
✔  **Didn't change the pictures to help it learn**
✔  **Didn't let it really learn about plant leaves**
✔  **Plant leaves look different than what it's used to (from ImageNet)**

---

# 📘 **Conclusion**

*   **SVM + HOG** is a good starting point, but can't handle lots of pictures.
*   **The basic CNN** did the best (84%) because it could learn what was important from the pictures.
*   **ResNet50** didn't do well because I didn't train it enough, and the pictures weren't great.

Basically:

👉 **The basic CNN** is the best for this project since the dataset size, how I processed the data, and the picture quality made it more suitable for this model.

---

# 📚 **References**

1.  PlantVillage Dataset, Hughes &amp; Salathé, Kaggle, 2015
2.  Kaiming He, Zhang, Ren &amp; Sun – *Deep Residual Learning for Image Recognition*, CVPR 2016
3.  Dalal &amp; Triggs – *Histograms of Oriented Gradients*, CVPR 2005
4.  François Chollet – *Deep Learning with Python*, Manning Publications
