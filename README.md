# 🎓 Student GPA Prediction using MLP (Multi-Layer Perceptron)

Predicting **Student Post Semester GPA** using a **Multi-Layer Perceptron (MLP)** built with **PyTorch**.

This project explores how various academic, personal, and AI-related factors influence a student's GPA. It is my **first Deep Learning project**, created to gain hands-on experience with Neural Networks, PyTorch, and the complete deep learning workflow—from data preprocessing to model training and evaluation.


---

## 📖 Project Overview

This project uses a **Multi-Layer Perceptron (MLP)** to predict a student's **Post Semester GPA** based on features from the **AI Impact on Students** dataset.

Unlike traditional machine learning algorithms, this project leverages a fully connected neural network capable of learning complex, non-linear relationships within the data.

The primary goal of this project was not only to build an accurate regression model but also to understand the fundamentals of Deep Learning by implementing every stage of the pipeline using **PyTorch**.

---

## ✨ Features

- Built a Deep Learning Regression Model using PyTorch
- End-to-End Neural Network Pipeline
- Exploratory Data Analysis (EDA)
- Outlier Detection & Removal
- Feature Encoding
- Feature Scaling
- Custom Multi-Layer Perceptron Architecture
- Model Training using Adam Optimizer
- Regression Performance Evaluation
- Modular Project Workflow

---

## 📊 Dataset

This project uses the **AI Impact on Students** dataset available on Kaggle.

**Source:**  
https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students

---

## 🛠️ Tech Stack

### Languages

- Python

### Libraries & Frameworks

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- PyTorch
- Jupyter Notebook

### Concepts

- Machine Learning
- Deep Learning
- Data Analysis
- Data Preprocessing
- Neural Networks
- Regression

---

# 📂 Project Workflow

```text
Dataset
   │
   ▼
Analysis.ipynb
(EDA)
   │
   ▼
Data_Preprocessing.ipynb
• Outlier Detection
• Feature Encoding
• Generate Preprocessed Dataset
   │
   ▼
preprocessed_data.csv
   │
   ▼
Main.py
• Feature Scaling
• Train-Test Split
• TensorDataset
• DataLoader
• Model Training
• Evaluation
• Saving Model
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Outlier Detection & Removal
- Encoding Categorical Features
- Feature Scaling
- Train-Test Split
- Tensor Conversion for PyTorch

---

# 📈 Exploratory Data Analysis

The complete EDA is available in **Analysis.ipynb**.

It includes:

- Univariate Analysis
- Bivariate Analysis
- Categorical Analysis
- Multivariate Analysis

---

# 🧠 Neural Network Architecture

The model is implemented using a **Multi-Layer Perceptron (MLP)**.

```text
Input Layer
      │
      ▼
Linear (32 → 64)
      │
     ReLU
      ▼
Linear (64 → 64)
      │
     ReLU
      ▼
Linear (64 → 32)
      │
     ReLU
      ▼
Linear (32 → 16)
      │
     ReLU
      ▼
Linear (16 → 1)
      │
      ▼
Predicted GPA
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Activation Function | ReLU |
| Loss Function | MSELoss |
| Optimizer | Adam |
| Batch Size | 32 |
| Epochs | 100 |

---

# ⚙️ Training Pipeline

The model follows the standard Deep Learning training workflow:

```text
TensorDataset
      │
      ▼
DataLoader
      │
      ▼
Forward Pass
      │
      ▼
Loss Calculation
      │
      ▼
Gradient Computation
      │
      ▼
Backpropagation
      │
      ▼
Optimizer Step
      │
      ▼
Repeat
```

---

# 📊 Model Performance

| Metric | Score |
|---------|------:|
| R² Score | 88.93% |
| MSE | 2.3787 |
| MAE | 12.2165 |
| RMSE | 15.4230 |

---

# 📁 Project Structure

```text
Student-GPA-Prediction/
│
├── Analysis.ipynb
├── Data_Preprocessing.ipynb
├── preprocessed_data.csv
├── Main.py
└── README.md
```

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Student-GPA-Prediction.git
```

Navigate to the project directory

```bash
cd Student-GPA-Prediction
```


---

# ▶️ Usage

### 1. Perform Exploratory Data Analysis

Run:

```text
Analysis.ipynb
```

---

### 2. Preprocess the Dataset

Run:

```text
Data_Preprocessing.ipynb
```

This notebook performs:

- Outlier Detection
- Feature Encoding
- Generates the preprocessed dataset

---

### 3. Train the Model

Run:

```bash
python Main.py
```

This script performs:

- Feature Scaling
- Train-Test Split
- Dataset Preparation
- DataLoader Creation
- Model Training
- Model Evaluation

---

# 📚 Acknowledgements

- Kaggle for providing the dataset.
- The PyTorch team for developing an excellent Deep Learning framework.
- The Scikit-Learn community for preprocessing and evaluation utilities.
- Open-source contributors whose libraries made this project possible.

---

# 👨‍💻 Author

**Kovi Venkata Hrushikesh**

💼 LinkedIn  
www.linkedin.com/in/kovihrushikesh

🌐 Portfolio  
hrushikeshkovi.netlify.app

---

## ⭐ If you found this project helpful, consider giving it a star!