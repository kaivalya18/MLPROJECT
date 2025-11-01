# Plant Disease Image Classification (Computer Vision)

## Description
The project uses and compares several machine learning models: SVM with HOG features, custom CNN and transfer learning with ResNet50 to classify plant diseases on leaf images in the PlantVillage dataset.  
Although SVM makes perfect accuracy in a small set, CNN model shows good performance in the entire test set with an accuracy of around 84%. The ResNet50 transfer learning version, however, performed poorly with a 34 percent accuracy, and so it is difficult to fine-tune or match the model and its output with the dataset in this specific experiment.



## Dataset Source
**PlantVillage Dataset:**  https://www.kaggle.com/datasets/emmarex/plantdisease
Open access that contains more than 54,000 disease-labeled crop leaf images. Supervised learning is conducted through the organization of images by the classes.



## Methods

### Data Preprocessing
- 64x64 pixel normalised images.  
- Encoded labels according to LabelEncoder.  
- Split of the dataset into train, validation and test with control of the classes balance.

### SVM + HOG
- Histogram of Oriented Gradients: the extracted features are those of a grayscale image.  
- A linear SVM based on a balanced 1500 sample subset was used to make the computationally efficient baseline.

### Custom CNN
- This is a convolutional network lightweight which has been trained in its entirety on the entire dataset and is accelerated by a GPU with an accuracy of 84%.

### Transfer Learning (ResNet50)
- Pretrained ResNet50 as a feature extractor with custom classification layers.  
- Trained over a few epochs but achieve comparatively low accuracy (=34%), maybe because of inadequate fine-tuning, image size bottlenecks or the complexity of the dataset.



## Step to Run

1. Clone the repository.  
2. Move and install the PlantVillage data in `data/` (or mount your Drive).  Dataset link- https://www.kaggle.com/datasets/emmarex/plantdisease
3. Install dependencies.  
4. Load run data, preprocess, train CNN and ResNet50 and evaluate with the supplied Jupyter notebooks or scripts.



## Experiments / Results Summary

| Model          | Accuracy (Test Set) | Observations |
|----------------|---------------------|---------------|
| SVM + HOG      | 80%                 | ok accuracy on 1500 sample subset. |
| Custom CNN     | 84%                 | Adequate speed and accuracy. |
| ResNet50 TL    | 34%                 | Lack of good performance, requires additional fine tuning. |



## Conclusion
SVM with hand-engineered HOG features obtains high results on a controlled subset but lacks scalability.  
The conventional CNN performs well of the bigger dataset and is the best option to this task.  
Theoretically promising transfer learning method based on ResNet50 needs further fine-tuning, data augmentation, or bigger images to work well in this application.


## References
- Salathé, M., Hughes, D.P. and PlantVillage Dataset, 2015, Kaggle Dataset.  
- K., Zhang, Ren, Sun, and Deep Residual Learning Image Recognition CVPR 2016.  
- Dalal, N., Triggs, B., Histograms of Oriented Gradients to human Detection, CVPR 2005.  
- Chollet, F., Deep Learning using Python, Manning Publications, 2017.
