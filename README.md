# 🚨 Fraud Detection using XGBoost 🚨

## 🌟 Overview

This project implements a **fraud detection system** using the **XGBoost machine learning algorithm**. It includes data preprocessing, model training, evaluation, and a **Flask web application** to allow users to test the model on new transactions.

The project is designed to demonstrate practical machine learning skills on tabular data and deploy a model for user interaction.

---

## ⚡ Features

* 🧹 Data cleaning and preprocessing
* ⚖️ Handling class imbalance (optional: SMOTE, class weights)
* 📊 XGBoost model for fraud detection
* 📈 Model evaluation using Precision, Recall, F1-Score, and AUC-ROC
* 🌐 Flask web application for testing predictions

---

## 📁 Folder Structure

```
fraud-detection-xgboost/
├── datasets/
│   └── Fraud.csv         # Placeholder for raw dataset (replace with actual file)
├── cleaned_data/
│   └── Fraud_cleaned.csv # Placeholder for cleaned dataset (replace with actual file)
├── models/
│   └── fraud_model.pkl   # Placeholder for trained model (replace with actual file)
├── templates/            # Flask HTML templates
│   └── index.html
├── static/               # Static files for Flask (CSS, JS)
├── clean_data.py         # Data preprocessing script
├── model_training.py     # Script to train XGBoost model
├── model_evaluation.py   # Script to evaluate model performance
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 📥 Dataset & Model Downloads

Due to GitHub file size limitations, the actual dataset and trained model are not included in the repository.

### Instructions:

1. Download the raw dataset from Google Drive: [Download Raw Dataset](https://drive.google.com/file/d/1daE_-CdjYlZLFRRXIVeLjOnb5h7AUvg6/view?usp=drive_link)
2. Place the file in the `datasets/` folder with the name `Fraud.csv`.
3. After preprocessing, download the cleaned dataset (optional): [Download Cleaned Dataset](https://drive.google.com/file/d/1N6Xl77ZHSm0eHDbgzkj3Re-PMoSDQXRm/view?usp=drive_link) and place it in `cleaned_data/` as `Fraud_cleaned.csv`.
4. Download the trained XGBoost model (optional): [Download Trained Model](https://drive.google.com/file/d/1in00l1XEVYR0GmRfa1evPdfX3oqs4tdx/view?usp=drive_link) and place it in `models/` as `fraud_model.pkl`.

---

## 🛠 Installation / Setup

1. Clone the repository:

```bash
git clone https://github.com/gauthamdv/fraud-detection-xgboost.git
cd fraud-detection-xgboost
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

* **Data Preprocessing:**

```bash
python clean_data.py
```

* **Model Training:**

```bash
python model_training.py
```

* **Model Evaluation:**

```bash
python model_evaluation.py
```

* **Flask App:**

```bash
python app.py
```

Then open the Flask app and **enter transaction details to test predictions**.

---

## 📊 Model Evaluation

The model performance is evaluated using:

* ✅ **Precision:** Measures how many predicted frauds are actually frauds.
* ✅ **Recall:** Measures how many actual frauds were correctly predicted.
* ✅ **F1-Score:** Harmonic mean of Precision and Recall.
* ✅ **AUC-ROC:** Measures overall model discrimination ability.

---

## 💡 Notes / Tips

* Ensure `datasets/Fraud.csv` and other placeholders are correctly named before running scripts.
* Large datasets or trained models are linked externally; download them before using the project.
* For contributions, make sure to include meaningful commit messages.

---

## 🤝 Contributions

Contributions are welcome! If you find bugs or want to improve the project, please submit a pull request or open an issue.
