# Brain Tumor MRI Classification with Vision Transformers

## Project Overview 
This project develops a **Vision Transformer (ViT) model** to classify brain MRI scans into four categories:  
`glioma` | `meningioma` | `pituitary` | `no tumor`  

**Key Objectives**:
- Achieve >95% validation accuracy for clinical viability  
- Minimize false negatives in pituitary cases (high mortality risk)  
- Identify common misclassification patterns  

## Data & Model Description
### Dataset ([Kaggle Source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- 7,043 MRI scans (5,712 training / 1,311 testing)
- **Class Distribution**:  
  ```python
  Training: glioma(1,321), meningioma(1,339), pituitary(1,457), no tumor(1,595)  
  Testing: glioma(300), meningioma(306), pituitary(300), no tumor(405)
  ```

### Model Architecture 
- Base Model: google/vit-base-patch16-224-in21k
- Fine-Tuning:
  ```python
  epochs: 3  
  batch_size: 16  
  image_size: 224x224  
  learning_rate: 3e-5
  ```
## Exploratory Data Analysis (EDA) Summary 
*Note: The EDA can be found in /notebooks.*
![image](https://github.com/user-attachments/assets/c80f63d3-5ba0-45ca-8cd9-ddecbc1aaea1)
- Imbalanced training set (20% more no-tumor cases) but balanced testing

![image](https://github.com/user-attachments/assets/7f8fdb2d-def4-4df4-a4d3-2fa49afc8388)
- Several key visual differences between glioma and meningioma classes

![image](https://github.com/user-attachments/assets/132d77c6-3a17-4627-ad0e-81b3f569daec)
- Optimal performance at Epoch 2

![image](https://github.com/user-attachments/assets/60b26f45-d483-4f08-ac62-4f2a983a34ae)
- Meningioma→Glioma errors are resolved, worsening Glioma→Meningioma errors

## Model Evaluation 
  ```python
| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Glioma     | 0.996     | 0.991  | 0.994 |
| Meningioma | 0.974     | 0.987  | 0.980 |
| Pituitary  | 0.909     | 1.000  | 0.952 |
| No Tumor   | 0.995     | 0.978  | 0.986 |
```
**Key Findings**
- 98.6% validation accuracy (exceeds clinical benchmarks)
- 100% recall for pituitary tumors (critical for diagnosis)
- Primary confusion: Glioma→Meningioma (11 cases in Epoch 3)

## Tableau Dashboard Features 
*Note: The tableau dashboard can be found in /Tableau Dashboard.*

- At-a-Glance KPIs: The dashboard leads with high-level Key Performance Indicators (KPIs), including the final Testing Accuracy and the model's Average Confidence Score, allowing for an immediate understanding of overall performance.
- Training vs. Testing Analysis: A dedicated bar chart compares the model's Accuracy and F1-Score on both the training and testing datasets. This feature is crucial for quickly diagnosing potential overfitting.
- Detailed Class-Level Metrics: To move beyond aggregate scores, a grouped bar chart breaks down the Precision, Recall, and F1-Score for each individual tumor class (Glioma, Meningioma, Pituitary, and No Tumor). This allows for the precise identification of classes where the model excels or struggles.
- Temporal Misclassification Tracking: An intuitive line chart tracks the evolution of specific error types (e.g., Glioma→Meningioma) across multiple training epochs. This visual uses direct labeling on each line for enhanced readability, eliminating the need for a separate legend and making trend analysis instantaneous.

## Conclusion & Next Steps
This exploratory analysis demonstrates that the ViT model achieves exceptional performance (98.6% accuracy) in classifying brain tumor MRI scans, despite inherent challenges:
- Class imbalance in training data (20% fewer glioma/meningioma samples)
- Textural similarities between glioma and meningioma (11 misclassified cases)
- Limited pituitary samples (though recall remains perfect)

**Theoretical Next Steps (for real-world deployment):**
  - Implement confidence thresholds for borderline predictions.
  - Augment with synthetic glioma-meningioma boundary cases.

*Note: While optimized for a portfolio project, the included recommendations highlight production-ready considerations.*  

## How to Run 
1. Clone the repository
```bash
git clone https://github.com/meerasanj/brain-tumor-ml-pipeline.git
cd brain-tumor-ml-pipeline
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Compile and run the script
```bash
python scripts/run_pipeline.py # to train the ViT model using 3 epochs
```
## License 

Copyright (c) 2025 Meera Sanjeevirao

This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
