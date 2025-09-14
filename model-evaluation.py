import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

class FraudModelEvaluator:
    def __init__(self, data_path, model_path="Models/xgb_fraud_model.pkl"):
        self.data_path = "Datasets/" + data_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None

    def load_data_and_model(self):
        self.df = pd.read_csv(self.data_path)
        self.model = joblib.load(self.model_path)
        print("Data and model loaded successfully.")

    def prepare_test_data(self, test_size=0.3, random_state=42):
        X = self.df.drop('isFraud', axis=1)
        y = self.df['isFraud']
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred))
        print("ROC AUC Score:", roc_auc_score(self.y_test, self.y_pred_proba))
        print("Average Precision (PR AUC):", average_precision_score(self.y_test, self.y_pred_proba))
        print("F1 Score:", f1_score(self.y_test, self.y_pred))

        cm = confusion_matrix(self.y_test, self.y_pred)
        self.plot_confusion_matrix(cm)
        self.plot_precision_recall_curve()

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

    def run_evaluation(self):
        self.load_data_and_model()
        self.prepare_test_data()
        self.evaluate_model()


if __name__ == "__main__":
    evaluator = FraudModelEvaluator("Fraud_cleaned.csv")
    evaluator.run_evaluation()
