# MLPROJECT

# Plant Disease Image Classification (Computer Vision)

## Description
This project implements and compares multiple machine learning models — SVM with HOG features, custom CNN, and transfer learning using ResNet50 — for classifying plant diseases from leaf images in the PlantVillage dataset.  
While SVM achieves perfect accuracy on a small subset, the CNN model demonstrates solid performance on the full test set with an accuracy of approximately 84%. However, the ResNet50 transfer learning model underperformed with an accuracy around 34%, indicating challenges in fine-tuning or dataset compatibility in this particular experiment.

## Dataset Source
- PlantVillage Dataset: Public dataset with over 54,000 images of crop leaves labeled by disease. Images are organized by class for supervised learning.

## Methods
- **Data Preprocessing:** Images resized to 64x64 pixels and normalized. Labels encoded using LabelEncoder. Dataset split into train, validation, and test sets maintaining class balance.
- **SVM + HOG:** Histogram of Oriented Gradients features extracted from grayscale images; a linear SVM trained on a balanced 1500-sample subset to provide a computationally efficient baseline.  
- **Custom CNN:** A lightweight convolutional network trained end-to-end on the full dataset with GPU acceleration, yielding an accuracy of 84%.  
- **Transfer Learning (ResNet50):** Pretrained ResNet50 used as a feature extractor with custom classification layers; trained for a few epochs but resulted in relatively low accuracy (~34%), possibly due to insufficient fine-tuning, image size constraints, or dataset complexity.

## Steps to Run
1. Clone the repository.  
2. Download and place the PlantVillage dataset in `data/` (or mount your Drive).  
3. Install dependencies:  


4. Run data loading, preprocessing, train CNN and ResNet50 models, and evaluate using the provided Jupyter notebooks or scripts.

## Experiments/Results Summary
| Model      | Accuracy (Test Set) | Observations                          |
|------------|---------------------|-------------------------------------|
| SVM + HOG  | 100%                | Perfect accuracy on 1500-sample subset |
| Custom CNN | 84%                 | Good balance of speed and accuracy  |
| ResNet50 TL| 34%                 | Poor performance, needs further tuning  |

## Conclusion
SVM combined with hand-engineered HOG features achieves excellent results on a controlled subset but is limited in scalability. The custom CNN offers a strong performance on the larger dataset and is an optimal choice for this task. The transfer learning approach using ResNet50, although promising theoretically, requires additional fine-tuning, data augmentation, or larger image sizes to improve performance in this use case.

## References
1. Hughes, D.P. and Salathé, M., PlantVillage Dataset, 2015, [Kaggle Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
2. He, K., Zhang, X., Ren, S., Sun, J., Deep Residual Learning for Image Recognition, CVPR 2016.  
3. Dalal, N., Triggs, B., Histograms of Oriented Gradients for Human Detection, CVPR 2005.  
4. Chollet, F., Deep Learning with Python, Manning Publications, 2017.

