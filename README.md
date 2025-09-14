🚨 Fraud Detection using XGBoost 🚨
🌟 Overview
This project implements a fraud detection system using the XGBoost machine learning algorithm. It includes data preprocessing, model training, evaluation, and a Flask web application to allow users to test the model on new transactions.
The project is designed to demonstrate practical machine learning skills on tabular data and deploy a model for user interaction.
________________________________________
⚡ Features
•	🧹 Data cleaning and preprocessing
•	⚖️ Handling class imbalance (optional: SMOTE, class weights)
•	📊 XGBoost model for fraud detection
•	📈 Model evaluation using Precision, Recall, F1-Score, and AUC-ROC
•	🌐 Flask web application for testing predictions
________________________________________
📁 Folder Structure
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
________________________________________
📥 Dataset & Model Downloads
Due to GitHub file size limitations, the actual dataset and trained model are not included in the repository.
Instructions:
1.	Download the raw dataset from Google Drive: Download Raw Dataset
2.	Place the file in the datasets/ folder with the name Fraud.csv.
3.	After preprocessing, download the cleaned dataset (optional): Download Cleaned Dataset and place it in cleaned_data/ as Fraud_cleaned.csv.
4.	Download the trained XGBoost model (optional): Download Trained Model and place it in models/ as fraud_model.pkl.
________________________________________
🛠 Installation / Setup
1.	Clone the repository:
git clone https://github.com/gauthamdv/fraud-detection-xgboost.git
cd fraud-detection-xgboost
2.	Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3.	Install dependencies:
pip install -r requirements.txt
________________________________________
🚀 Usage
•	Data Preprocessing:
python clean_data.py
•	Model Training:
python model_training.py
•	Model Evaluation:
python model_evaluation.py
•	Flask App:
python app.py
Enter transaction details to test predictions.
________________________________________
📊 Model Evaluation
The model performance is evaluated using:
•	✅ Precision: Measures how many predicted frauds are actually frauds.
•	✅ Recall: Measures how many actual frauds were correctly predicted.
•	✅ F1-Score: Harmonic mean of Precision and Recall.
•	✅ AUC-ROC: Measures overall model discrimination ability.
________________________________________
💡 Notes / Tips
•	Ensure datasets/Fraud.csv and other placeholders are correctly named before running scripts.
•	Large datasets or trained models are linked externally; download them before using the project.
•	For contributions, make sure to include meaningful commit messages.
________________________________________
🤝 Contributions
Contributions are welcome! If you find bugs or want to improve the project, please submit a pull request or open an issue.

