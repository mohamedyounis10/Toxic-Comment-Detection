<div align="center">
  <h1>Toxic Comment Detection 🛡️🤖💙</h1>
  <p>End-to-end NLP pipeline for multi-label toxicity classification (cleaning 🧹 → feature engineering ⚙️ → baseline ML 📈 → DistilBERT fine-tuning 🧠 → deployment 🚀).</p>

  <p>
    <a href="#overview">Overview 🧾</a> •
    <a href="#business-problem">Business Problem 🎯</a> •
    <a href="#project-structure">Project Structure 🗂️</a> •
    <a href="#dataset">Dataset 🧩</a> •
    <a href="#workflow">Workflow 📒</a> •
    <a href="#results">Results 📊</a> •
    <a href="#challenges">Challenges ⚠️</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue" />
    <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
    <img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-Baseline_ML-F7931E" />
    <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-Transformers-FFBF00" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C" />
    <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B" />
  </p>
</div>

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Business Problem 🎯](#business-problem)
- [Project Structure 🗂️](#project-structure)
- [Dataset 🧩](#dataset)
- [Workflow 📒](#workflow)
- [Results 📊](#results)
- [Challenges ⚠️](#challenges)
- [Streamlit Apps 🖼️](#streamlit-apps)
- [How to Run ▶️](#how-to-run)
- [Author ✍️](#author)

---

<a id="overview"></a>
## Overview 🧾

This repository focuses on **multi-label toxic comment detection**, combining classical ML and Transformer-based NLP:

- **Data Cleaning & Normalization 🧹**: text preprocessing, regex cleanup, lemmatization, and stopword handling.
- **Baseline Modeling 📈**: TF-IDF + weighted classical model for explainable and lightweight inference.
- **Deep Learning Modeling 🧠**: DistilBERT fine-tuning for stronger contextual understanding.
- **Model Serving 🚀**: interactive Streamlit apps for real-time toxic speech analysis.
- **Multi-label Outputs 🏷️**: detect 6 toxicity classes in a single prediction.

---

<a id="business-problem"></a>
## Business Problem 🎯

Online communities and platforms need automatic moderation to reduce harmful content.

Without reliable toxicity detection:

- Harmful language can spread quickly and damage user trust 😞
- Moderation teams face high manual workload and delayed responses ⏰
- Platform safety and user retention are negatively impacted 📉

Goal: classify each comment across six toxic categories to support proactive moderation and safer user experiences 🤝

---

<a id="project-structure"></a>
## Project Structure 🗂️

The project has two main tracks:

- **`Basemodel/`** - classical NLP baseline (TF-IDF + weighted model)
- **`DistilBERT/`** - Transformer fine-tuned approach

Tree 🌳:

```text
Project/
├─ Basemodel/
│  ├─ Dataset/
│  │  ├─ train.csv
│  │  └─ test.csv
│  ├─ Dataset_cleaned/
│  │  ├─ cleaned_train_data.csv
│  │  └─ cleaned_test_data.csv
│  ├─ Models/
│  │  ├─ tfidf_vectorizer.pkl
│  │  └─ toxic_weighted_model.pkl
│  ├─ notebook.ipynb
│  ├─ app.py
│  └─ examples.txt
├─ DistilBERT/
│  ├─ notebook-finetuning.ipynb
│  ├─ application.py
│  └─ toxic_model_zip/
│     ├─ config.json
│     ├─ model.safetensors
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     └─ training_args.bin
└─ README.md
```

---

<a id="dataset"></a>
## Dataset 🧩

The dataset is used for **multi-label toxicity classification**.

Target labels 🎯:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Each comment can belong to one or more classes at the same time.

---

<a id="workflow"></a>
## Workflow 📒

### 1) Baseline Track (`Basemodel`) ⚙️

1. Load and inspect raw train/test data
2. Clean text (regex normalization + NLP preprocessing)
3. Build TF-IDF features
4. Train weighted multi-label classifier
5. Save artifacts:
   - `Models/tfidf_vectorizer.pkl`
   - `Models/toxic_weighted_model.pkl`
6. Serve predictions with Streamlit (`Basemodel/app.py`)

### 2) DistilBERT Track (`DistilBERT`) 🧠

1. Fine-tune DistilBERT on toxicity labels
2. Export model/tokenizer files to `toxic_model_zip/`
3. Run advanced Streamlit inference app (`DistilBERT/application.py`)
4. Show class probabilities and tokenization breakdown

---

<a id="results"></a>
## Results 📊

### Baseline Model ✅

- Fast and lightweight inference using TF-IDF features.
- Useful explainability via preprocessing and score-based outputs.

### DistilBERT Model 🏆

- Better contextual understanding for toxic language.
- Produces probability scores per label for robust moderation decisions.

---

<a id="challenges"></a>
## Challenges ⚠️

- **Noisy user-generated text**: slang, misspellings, symbols, and mixed writing styles.
- **Label imbalance**: rare classes (e.g., `threat`) are harder to learn.
- **Multi-label complexity**: one comment may trigger multiple toxicity categories.
- **Deployment trade-off**: baseline is faster, DistilBERT is typically stronger contextually.

These were addressed through careful preprocessing, weighted learning, and a dual-model strategy 💡

---

<a id="streamlit-apps"></a>
## Streamlit Apps 🖼️

### Baseline App
```bash
streamlit run Basemodel/app.py
```

### DistilBERT App
```bash
streamlit run DistilBERT/application.py
```

---

<a id="how-to-run"></a>
## How to Run ▶️

### 1) Setup Environment 🧪
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install Dependencies 📦
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib spacy torch transformers jupyter
python -m spacy download en_core_web_sm
```

### 3) Run Notebooks 🚀
```bash
jupyter notebook Basemodel/notebook.ipynb
```

```bash
jupyter notebook DistilBERT/notebook-finetuning.ipynb
```

---

<a id="author"></a>
## Author ✍️

- **Name**: Mohamed Younis
- **Track**: NLP Project
<div align="center">
  <h1>Customer Churn Prediction 📉🤖💙</h1>
  <p>End-to-end churn analysis and modeling (EDA 🔍 → preprocessing 🧹 → modeling ⚙️ → best model export 💾) using Python.</p>

  <p>
    <a href="#overview">Overview 🧾</a> •
    <a href="#business-problem">Business Problem 🎯</a> •
    <a href="#project-structure">Project Structure 🗂️</a> •
    <a href="#dataset">Dataset 🧩</a> •
    <a href="#notebook-journey">Notebook Journey 📒</a> •
    <a href="#results">Results 📊</a> •
    <a href="#challenges">Challenges ⚠️</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue" />
    <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
    <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data_Analysis-150458" />
    <img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E" />
    <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Modeling-EC6B23" />
  </p>
</div>

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Business Problem 🎯](#business-problem)
- [Project Structure 🗂️](#project-structure)
- [Dataset 🧩](#dataset)
- [Notebook Journey 📒](#notebook-journey)
- [Results 📊](#results)
- [Challenges ⚠️](#challenges)
- [Streamlit Preview 🖼️](#streamlit-preview)
- [How to Run ▶️](#how-to-run)
- [Author ✍️](#author)

---

<a id="overview"></a>
## Overview 🧾

This repository focuses on **Customer Churn Prediction**, building a full business-oriented machine learning workflow:

- **Data Understanding 🔍**: Explore train/test behavior and target distribution.
- **Data Preparation 🧹**: Clean, align, and prepare data for modeling.
- **Feature Engineering ⚙️**: Build stronger behavioral features to improve robustness.
- **Modeling & Comparison 📈**: Train and compare multiple models fairly.
- **Model Export 💾**: Automatically save the best model for later use.

---

<a id="business-problem"></a>
## Business Problem 🎯

The company needs to reduce customer churn and retain high-value customers.

Without an early warning model:

- Retention campaigns become expensive and less targeted 💸
- Teams react too late to prevent churn ⏰
- Revenue and CLV are negatively impacted 📉

Goal: predict whether a customer will churn (`Churn = 1`) to enable proactive retention actions 🤝

---

<a id="project-structure"></a>
## Project Structure 🗂️

- [`Datasets/` 🧩](#dataset) - project datasets
  - `train.csv`
  - `test.csv`
- [`notebook.ipynb` 📒](#notebook-journey) - full analysis and modeling workflow
- `README.md` - project documentation
- `best_model.pkl` - exported best model (generated after running final notebook cells)

Tree 🌳:

```text
Project/
├─ Datasets/
│  ├─ train.csv
│  └─ test.csv
├─ notebook.ipynb
├─ README.md
├─ Presentation.pdf
├─ image.png
├─ app.py
└─ best_model.pkl
```

---

<a id="dataset"></a>
## Dataset 🧩

The project uses customer-level records for churn classification.

Target Column 🎯:
- **`Churn`**: Binary target (`0 = No Churn`, `1 = Churn`)

Example feature groups 🧾:
- Customer profile (demographic-like fields)
- Usage behavior (frequency/intensity indicators)
- Service experience (e.g., support interactions)
- Payment/spend behavior (delay and spending patterns)

---

<a id="notebook-journey"></a>
## Notebook Journey 📒

The notebook is organized as a clear end-to-end flow:

1. **Required Libraries** 📦
2. **Read the Dataset** 📥
3. **Exploratory Data Analysis (EDA)** 🔍
   - info, describe, missing values, duplicates, target checks
   - numeric/categorical visual analysis
   - churn-rate exploration by groups
4. **Data Preprocessing** 🧹
   - drop unnecessary columns
   - categorical encoding
   - feature engineering (`Usage_Per_Tenure`)
5. **Split Features/Target** ✂️
6. **Train/Test Split** ⚙️
7. **Modeling** 🤖
   - Random Forest
   - Logistic Regression
   - XGBoost
8. **Modeling Summary & Best Model Selection** 🏆
   - compare models on key churn metrics
   - save best model to `best_model.pkl`

---

<a id="results"></a>
## Results 📊

### Modeling Output ✅

- All candidate models are trained and evaluated.
- A final summary table compares:
  - `Accuracy`
  - `Precision (Churn=1)`
  - `Recall (Churn=1)`
  - `F1 (Churn=1)`

### Best Model Selection 🏆

- The best model is selected based on the highest **`F1 (Churn=1)`**.
- Final artifact is saved as:
  - `best_model.pkl`

---

<a id="challenges"></a>
## Challenges ⚠️

- **Data consistency issues**: required cleaning and type handling.
- **Domain shift risk**: noticeable distribution differences between train/test in some features.
- **Metric trade-off**: balancing precision and recall is critical in churn use cases.

These challenges are addressed through careful EDA, feature engineering, and model comparison strategy 💡

---

<a id="streamlit-preview"></a>
## Streamlit Preview 🖼️

<img width="1919" height="885" alt="Screenshot 2026-04-30 105140" src="https://github.com/user-attachments/assets/c7d5bb81-15c3-4582-8fc0-5af2defee133" />

```bash
streamlit run app.py
```

---

<a id="how-to-run"></a>
## How to Run ▶️

### 1) Setup Environment 🧪
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install Dependencies 📦
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter streamlit
```

### 3) Launch Notebook 🚀
```bash
jupyter notebook notebook.ipynb
```

Run all cells in order, then verify `best_model.pkl` is generated.

---

<a id="author"></a>
## Author ✍️

- **Name**: Mohamed Younis
- **Program**: MSC KFS - Data Science Phase 2 💙
