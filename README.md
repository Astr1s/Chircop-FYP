## 📘 Project Overview

This project focuses on creating and evaluating graph-based models for bug triage. Below are the steps to get started with using this repository.

---

## 📥 Setup Instructions

### 1. Clone the Repository

Clone the repository using the following commands:

git clone https://github.com/<your-username>/Chircop-FYP.git  
cd Chircop-FYP

---

### 2. Prepare the Raw Data

Place your raw data inside a folder of your choice. This raw data will be processed in the next step.

---

### 3. Generate the Graph Data

Navigate to the Data_Graph_Setup directory. You will find three key files:

- DataPreprocessEclipse.py  
- DataPreprocessMozilla.py  
- graphsetupBERT.py  

Steps:  
1. Run either DataPreprocessEclipse.py or DataPreprocessMozilla.py depending on the dataset you are using.  
2. These scripts will generate three .pkl files.  
3. Use the generated .pkl files as input to graphsetupBERT.py to create the graph required for training.

---

### 4. Train the Models

Go to the Training_Models directory. It contains five Python scripts that train different models using the graph created in the previous step.

---

### 5. Evaluate the Models

The Evaluation folder contains all evaluation tools. It includes a subfolder named reTrain, which allows retraining the models on a separate evaluation graph.

Key Files:  
- reTrain/ — Retrains models using the evaluation graph.  
- DataEval.py — Computes statistics and dataset insights.  
- eval.py — Calculates Balanced Top-K Accuracy and Mean Reciprocal Rank (MRR).  
- furthereval.py — Generates additional figures used in the dissertation.
