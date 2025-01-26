# NeuroLens

---

## 📌 Overview
**NeuroLens** is a deep learning project designed to classify brain tumor MRI scans into four categories: **glioma**, **meningioma**, **notumor**, and **pituitary**. Built with TensorFlow and Keras, this project leverages a pre-trained ResNet50 model, enhanced with custom layers and advanced techniques like mixed-precision training and data augmentation. The goal is to provide a robust framework for medical image analysis, aiding in early and accurate tumor detection.

---

## 🧠 Dataset
The dataset is sourced from Kaggle and contains **MRI images** organized into four classes:
- **Glioma**: 1,321 training images | 300 testing images  
- **Meningioma**: 1,339 training images | 306 testing images  
- **Notumor**: 1,595 training images | 405 testing images  
- **Pituitary**: 1,457 training images | 300 testing images  

**Structure**:  
```
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🛠️ Key Features
1. **Advanced Preprocessing**  
   - **Bilateral Filtering**: Reduces noise while preserving edges.
   - **Color Mapping**: Enhances contrast using `COLORMAP_BONE`.
   - **Resizing**: Standardizes images to **224x224 pixels** for ResNet50 compatibility.
   - **Normalization**: Scales pixel values to `[0, 1]`.

2. **Data Augmentation**  
   Techniques include:
   - Random rotation (±12°)
   - Width/height shifting (±7%)
   - Horizontal flipping
   - Reflect padding

3. **Model Architecture**  
   - **Base Model**: ResNet50 (pre-trained on ImageNet) with frozen weights.
   - **Custom Layers**:
     - Global Average Pooling
     - Dropout (40% rate)
     - Dense layer with L2 regularization (λ=0.001) and softmax activation.

4. **Training Optimization**  
   - **Mixed Precision Training**: Uses `mixed_float16` for faster GPU computation.
   - **Callbacks**:
     - Early stopping (patience=5)
     - Learning rate reduction on plateau (factor=0.3)
     - Model checkpointing (saves best model by validation loss)
     - TensorBoard integration for real-time metrics.

---

## 📊 Performance Metrics
### Training Configuration
- **Optimizer**: AdamW (learning rate = 0.0001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 20 (training), 64 (validation)
- **Epochs**: 15

### Results
- **Training Accuracy**: 98.5% (example)
- **Validation Accuracy**: 94.2% (example)
- **Test Accuracy**: 92.8% (example)

#### Confusion Matrix  
<div align="center">
  <img src="https://github.com/user-attachments/assets/89dc7c48-4633-432c-ac78-531d3d6748e8" alt="Confusion Matrix" width="50%" style="display: block; margin: 0 auto;">
  <p><em>Normalized confusion matrix showing class-wise precision and recall.</em></p>
</div>

#### Loss and Accuracy Curves  
![Training Curves](https://via.placeholder.com/800x400?text=Loss+and+Accuracy+Curves)  
*Training vs. validation loss and accuracy over epochs.*

---

## 🚀 Usage
### Prerequisites
- Python 3.10+
- GPU with CUDA support (recommended)
- Libraries: TensorFlow, OpenCV, Pandas, Matplotlib, Scikit-learn

### Steps to Reproduce
1. **Install Dependencies**:
   ```bash
   pip install tensorflow opencv-python pandas matplotlib scikit-learn kagglehub
   ```
2. **Download Dataset**:
   ```python
   import kagglehub
   dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
   ```
3. **Run the Notebook**: Execute `NeuroLens.ipynb` to preprocess data, train the model, and evaluate performance.

---

## 📂 Project Structure
```
NeuroLens/
├── NeuroLens.ipynb          # Main Jupyter notebook
├── logs/                    # TensorBoard logs
├── saved_models/            # Best model checkpoints
├── requirements.txt         # Dependency list
└── README.md                # This file
```

---

## 🔍 Insights and Future Work
### Insights
- The model achieves high accuracy but shows slight overfitting (gap between training and validation accuracy).
- **Pituitary** and **notumor** classes have higher recall, while **glioma** and **meningioma** may require more data.

### Future Improvements
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and regularization.
- **Advanced Architectures**: Test EfficientNet or Vision Transformers (ViTs).
- **Class Balancing**: Address dataset imbalance using oversampling or weighted loss.
- **Explainability**: Integrate Grad-CAM for visual explanations of predictions.

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**NeuroLens** aims to bridge the gap between deep learning and medical diagnostics, providing a scalable solution for brain tumor classification. Contributions and feedback are welcome! 🌟
