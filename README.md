# Brain Tumor MRI Classification with Vision Transformers

## Project Overview 
This project leverages a Vision Transformer (ViT) model to build an end-to-end pipeline for medical image analysis. The model is fine-tuned to classify brain MRI scans into four distinct and clinically relevant categories:  `glioma` | `meningioma` | `pituitary` | `no tumor`. The objective is to demonstrate how modern AI architectures can achieve high-accuracy classification, serving as a powerful decision support tool in a medical setting.

**Key Objectives**:
- Achieve >95% validation accuracy for clinical viability  
- Minimize false negatives in pituitary cases (high mortality risk)  
- Identify common misclassification patterns

**Tech Stack**
- Data Analysis & Manipulation: Python, Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn, Tableau
- Machine Learning & Deep Learning:
  - Hugging Face Transformers: To access the pre-trained Vision Transformer (ViT) model.
  -  PyTorch / TensorFlow: The backend framework for training and fine-tuning the model.
  - Scikit-learn: For generating the classification report and performance metrics.
  - Development Environment: Google Colab, Jupyter Notebook

## Model & Data Description
### Model Architecture:
- Hugging Face Base Model: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- Pre-training: The base model was pre-trained on ImageNet-21k (14 million images across 21k classes), learning a powerful visual representation by treating images as a sequence of 16x16 patches.
- Fine-Tuning:
  ```python
  epochs: 3  
  batch_size: 16  
  image_size: 224x224  
  learning_rate: 3e-5
  ```
  
### Dataset ([Kaggle Source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- 7,043 MRI scans (5,712 training / 1,311 testing)
- Class Distribution:  
  ```python
  Training: glioma(1,321), meningioma(1,339), pituitary(1,457), no tumor(1,595)  
  Testing: glioma(300), meningioma(306), pituitary(300), no tumor(405)
  ```

### Data Processing:

**1. Dataset Structure**:

Verified folder structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

**2. Image Transformations**

Applied to all images:
1. Resize: 256×256
2. Center Crop: 224×224 (ViT input size)
3. Normalize using [ImageNet stats](https://docs.pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize):  
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

**3. Data Loaders**

| Feature          | Training           | Testing           |
|------------------|--------------------|-------------------|
| **Shuffle**      | Yes                | No                |
| **Batch Size**   | Configurable (16)  | Configurable (16) |
| **Workers**      | 2                  | 2                 |

**Key Notes**:
- Automatic label assignment via folder names
- On-the-fly transformations
- RGB conversion enforced
- Identical preprocessing for training/inference

## Exploratory Data Analysis (EDA) Summary 
*Note: The EDA can be found in `/notebooks`.*
![image](https://github.com/user-attachments/assets/c80f63d3-5ba0-45ca-8cd9-ddecbc1aaea1)
- Imbalanced training set (20% more no-tumor cases) but balanced testing set

![image](https://github.com/user-attachments/assets/7f8fdb2d-def4-4df4-a4d3-2fa49afc8388)
- Several key visual differences between glioma and meningioma classes

![image](https://github.com/user-attachments/assets/132d77c6-3a17-4627-ad0e-81b3f569daec)
- Optimal performance at Epoch 2

![image](https://github.com/user-attachments/assets/60b26f45-d483-4f08-ac62-4f2a983a34ae)
- Meningioma→Glioma errors are resolved, worsening Glioma→Meningioma errors

## Model Evaluation 
### Performance Benchmarking:

**Initial Untrained Performance:**
  ```python
Initial Accuracy (before training): 25.04%  
Confusion Matrix:  
[[   3  218   11    0]  
 [   3  144    2    0]  
 [   0   20    0    0]  
 [   0  186    0    0]]
  ```
- Matches random chance (25% for 4-class problem)
- Model initially biased toward meningioma predictions

**Final Trained Performance & Confusion Matrix:**
  ```python
| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Glioma     | 0.996     | 0.991  | 0.994 |
| Meningioma | 0.974     | 0.987  | 0.980 |
| Pituitary  | 0.909     | 1.000  | 0.952 |
| No Tumor   | 0.995     | 0.978  | 0.986 |
```

![image](https://github.com/user-attachments/assets/cf626e96-11a1-4ebd-a677-cd323bf52d1f)

**Key Findings**
- 98.6% validation accuracy (exceeds clinical benchmarks)
- 100% recall for pituitary tumors (critical for diagnosis)
- Primary confusion: Glioma→Meningioma (11 cases in Epoch 3)

## Tableau Dashboard Features 
**Note:** The Tableau dashboard can be found in `/Tableau Dashboard` or directly on [Tableau Public](https://public.tableau.com/views/BrainScanProjectDashboard/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

- At-a-Glance KPIs: The dashboard leads with high-level Key Performance Indicators (KPIs), including the final Testing Accuracy and the model's Average Confidence Score, allowing for an immediate understanding of overall performance.
- Training vs. Testing Analysis: A bar chart compares the model's Accuracy and F1-Score on both the training and testing datasets. This feature is crucial for quickly diagnosing potential overfitting.
- Class-Level Metrics: To move beyond aggregate scores, a grouped bar chart breaks down the Precision, Recall, and F1-Score for each individual tumor class (Glioma, Meningioma, Pituitary, and No Tumor). This allows for the precise identification of classes where the model excels or struggles.
- Misclassification Tracking: A line chart tracks the evolution of specific error types (e.g., Glioma→Meningioma) across multiple training epochs. This visual uses direct labeling on each line for enhanced readability, eliminating the need for a separate legend and making trend analysis instantaneous.

## Conclusion & Next Steps
This exploratory analysis demonstrates that the ViT model achieves exceptional performance (98.6% accuracy) compared to the initial performance (25.04% accuracy) in classifying brain tumor MRI scans, despite inherent challenges:
- Class imbalance in training data (20% fewer glioma/meningioma samples)
- Textural similarities between glioma and meningioma (11 misclassified cases)
- Limited pituitary samples (though recall remains perfect)

**Theoretical Next Steps (for real-world deployment):**
  - Implement confidence thresholds for borderline predictions.
  - Augment with synthetic glioma-meningioma boundary cases.

*Note: While optimized for a portfolio project, the included recommendations highlight production-ready considerations.*  

## How to Run 
1. Clone the repository
```
git clone https://github.com/meerasanj/brain-tumor-ml-pipeline.git
cd brain-tumor-ml-pipeline
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the script
```
# This command trains the ViT model using the default 3 epochs
python scripts/run_pipeline.py
```
## License 

Copyright (c) 2025 Meera Sanjeevirao

This use of the ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)) dataset is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).

### Acknowledgements & Citations
This project utilized the pre-trained [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224-in21k) model from the Hugging Face library. The development of this model architecture and the dataset it was trained on are credited to the following research papers:

- For the Vision Transformer (ViT) model:
Wu, B., Xu, C., Dai, X., Wan, A., Zhang, P., Yan, Z., Tomizuka, M., Gonzalez, J., Keutzer, K., & Vajda, P. (2020). Visual Transformers: Token-based Image Representation and Processing for Computer Vision. arXiv preprint arXiv:2006.03677.

- For the ImageNet dataset (used for pre-training the ViT model):
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition.
