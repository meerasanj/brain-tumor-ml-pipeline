# Brain Tumor MRI Classification with Vision Transformers

## Project Overview 
This project develops a **Vision Transformer (ViT) model** to classify brain MRI scans into four categories:  
`glioma` | `meningioma` | `pituitary` | `no tumor`  

**Key Objectives**:
- Achieve >95% validation accuracy for clinical viability  
- Minimize false negatives in pituitary cases (high mortality risk)  
- Identify common misclassification patterns  

## Data & Model Description
### Dataset ([Source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- 7,043 MRI scans (5,712 training / 1,331 testing)
- **Class Distribution**:  
  ```python
  Training: glioma(1,321), meningioma(1,339), pituitary(1,457), no tumor(1,595)  
  Testing: glioma(300), meningioma(306), pituitary(300), no tumor(405)
  ```
  