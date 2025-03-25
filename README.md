# **🔬 Pneumonia Classification & Lung Segmentation using Deep Learning**
**Detecting Pneumonia in Chest X-Rays with CNNs & Segmentation Models**

## **📌 Project Overview**
Pneumonia is a serious lung infection requiring **early detection** for effective treatment. Manual diagnosis using **chest X-rays (DICOM images)** is time-consuming and prone to variability. This project leverages **deep learning** for:  

✅ **Classification:** Detecting pneumonia vs. normal lungs using CNNs.  
✅ **Segmentation:** Identifying lung regions and highlighting infected areas.  

### **🚀 Key Challenges**
- **Variability in X-ray quality** affecting classification accuracy.  
- **Similar appearance** of pneumonia with other lung diseases.  
- **Precise localization** of infected regions for segmentation.  

---

## **🗂 Dataset & Preprocessing**
📌 **Dataset:** *RSNA Pneumonia Detection Challenge (Kaggle, 2018)*  
- **30,227 chest X-ray images** with labeled pneumonia regions.  
- Images contain **0, 1, or multiple bounding boxes** marking infections.  

### **🔍 Preprocessing Steps**
- Converted **DICOM images** to **grayscale 8-bit** format.  
- Resized images to **1024×1024 resolution** for uniformity.  
- Applied **windowing & leveling** to enhance X-ray visibility.  
- Bounding box annotations were **visualized & extracted** for training.  

---

## **⚙️ Model Architectures**
### **1️⃣ Infection Classification (CNNs)**
#### **VGG16 (Trained from Scratch)**
- Custom **VGG16 CNN** trained on pneumonia dataset.  
- **Adam optimizer (LR = 1e-4)** to prevent large weight updates.  
- **Results:**  
  - Overfitting detected at **epoch 8**, requiring regularization.  
  - **Training Accuracy:** 85.2% | **Validation Accuracy:** 77.8%  

#### **ResNet50 (Pretrained)**
- **ResNet50V2 with ImageNet weights** used as a feature extractor.  
- Fine-tuned **final layers for pneumonia classification**.  
- **RandomRotation(0.15)** applied for augmentation.  
- **Results:**  
  - Fine-tuning **improved accuracy (77.87% → 78.09%)**.  
  - **Precision for Pneumonia:** **0.66 → 0.68** (reduced false positives).  

---

### **2️⃣ Lung Segmentation (U-Net & DeepLabV3+)**
#### **U-Net Architecture**
- Trained on **Montgomery County & Shenzhen Hospital datasets**.  
- Resized images to **512×512 pixels** for training.  
- **Dilation applied to masks** for better boundary detection.  
- **Results:**  
  - **Segmentation Accuracy:** 90%  
  - Overfitting detected in **early training epochs**.  

#### **DeepLabV3+ (Pretrained ResNet101)**
- **Advanced segmentation model** with **Atrous Spatial Pyramid Pooling (ASPP)**.  
- **Fine-tuned on pneumonia cases** from the RSNA dataset.  
- **Results:**  
  - **IoU (Intersection over Union):** 93% (better than U-Net).  
  - Improved boundary precision in **complex cases**.  

---

## **📊 Model Comparisons**
| Model         | Accuracy (%) | Precision | Recall | AUC  | Overfitting Issue? |
|--------------|-------------|-----------|--------|------|------------------|
| **VGG16**    | 77.8        | 0.65      | 0.72   | 0.76 | Yes (epoch 8)    |
| **ResNet50** | 78.1        | 0.68      | 0.74   | 0.78 | No               |

| Segmentation Model | IoU Score | Accuracy (%) | Notes |
|--------------------|----------|--------------|-------|
| **U-Net**         | 0.89     | 90%          | Fast, but less accurate on complex cases |
| **DeepLabV3+**    | 0.93     | 95%          | Better boundary precision & generalization |

---

## **🚀 Implementation**
### **1️⃣ Clone the Repository**
```bash
git clone <your-repo-url>
cd pneumonia-detection
