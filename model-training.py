import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

class FraudDetectionModel:
    def __init__(self, data_path, model_path="xgb_fraud_model.pkl"):
        self.data_path = "Datasets/" + data_path
        self.model_path = "Models/" + model_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess_data(self, test_size=0.3, random_state=42):
        X = self.df.drop('isFraud', axis=1)
        y = self.df['isFraud']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        scale_pos_weight = self.y_train.value_counts()[0] / self.y_train.value_counts()[1]

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=2.0,
            reg_alpha=0.5,
            gamma=1,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed successfully.")
        return self.model

    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved as {self.model_path}")
        else:
            print("No model to save. Train the model first.")

    def run_pipeline(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.save_model()


if __name__ == "__main__":
    model = FraudDetectionModel("Fraud_cleaned.csv")
    model.run_pipeline()
