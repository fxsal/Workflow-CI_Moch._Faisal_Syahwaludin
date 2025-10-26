import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# Inisialisasi DagsHub dan MLflow Online
dagshub.init(repo_owner='fxsal', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/fxsal/Membangun_model.mlflow")
mlflow.set_experiment("E-Commerce Shipping - Random Forest")

# Load Dataset hasil preprocessing
df = pd.read_csv('E-Commerce_Shipping_Data_preprocessing.csv')

# Pastikan nama kolom target sesuai dataset
X = df.drop(columns=['reached_on_time'])
y = df['reached_on_time']

# Standarisasi fitur numerik
num_cols = ['cost_of_the_product', 'discount_offered', 'weight_in_gms',
            'customer_rating', 'prior_purchases']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Penyeimbangan data menggunakan SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split data latih & uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Grid parameter optimal dan efisien
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [12, 15, 20],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Model Random Forest
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

# Hyperparameter tuning menggunakan GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Menjalankan eksperimen MLflow (Manual Logging)
with mlflow.start_run(run_name="RandomForest_Tuning") as run:
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Prediksi dan metrik evaluasi
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    balanced_acc = (rec + prec) / 2

    # Logging parameter hasil tuning
    mlflow.log_params(best_model.get_params())

    # Logging metrik performa
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_acc
    })

    # Tambahan metrik (syarat Advanced)
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_negative", cm[0][0])
    mlflow.log_metric("false_positive", cm[0][1])
    mlflow.log_metric("false_negative", cm[1][0])
    mlflow.log_metric("true_positive", cm[1][1])

    # Simpan artefak model ke DagsHub
    mlflow.sklearn.log_model(best_model, "RandomForest_Model")

print("Hyperparameter tuning & manual logging berhasil ke DagsHub.")
print("Lihat tracking di: https://dagshub.com/fxsal/Membangun_model.mlflow")