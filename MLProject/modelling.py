import os
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DAGSHUB_USER = "fxsal"
DAGSHUB_REPO = "membangun_model_ml"
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

dagshub.auth.add_app_token(DAGSHUB_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

# Inisialisasi DagsHub dan MLflow Online
dagshub.init(repo_owner='fxsal', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/fxsal/Membangun_model.mlflow")
mlflow.set_experiment("E-Commerce Shipping - Basic AutoLog")

# Load Dataset hasil preprocessing
df = pd.read_csv('MLProject/E-Commerce_Shipping_Data_preprocessing.csv')
X = df.drop(columns=['reached_on_time'])
y = df['reached_on_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training dan Autolog ke DagsHub
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

print("Model berhasil dilatih dan hasil dilog ke DagsHub.")

print("Lihat tracking di: https://dagshub.com/fxsal/Membangun_model.mlflow")




