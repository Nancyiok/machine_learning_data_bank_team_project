#  k-NN та Decision Tree гіперпараметри підібрані
#  Logistic Regression використано class_weight='balanced' через дисбаланс класів 88/12;
#  Random Forest (n_estimators=200) та HistGradientBoosting: базові параметри за замовчуванням.
#  Baseline: DummyClassifier, який завжди передбачає найчастіший клас ("Ні").
import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from sklearn.base import clone 


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_NAME = "Baseline (always No)"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_class.csv"))["Subscription"]

print("Розподіл класів у train (0 = Ні, 1 = Так):")
print(y_train.value_counts(normalize=True).round(3).to_string())
print("-" * 50)

# Найкращі параметри з попередніх скриптів
params_path = os.path.join(RESULTS_DIR, "params.json")
with open(params_path, encoding="utf-8") as f:
    params = json.load(f)

BEST_K = params["BEST_K"]
BEST_WEIGHTS = params["BEST_WEIGHTS"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_CRITERION = params["TREE_CRITERION"]
LR_CLASS_WEIGHT = "balanced"

print(f"KNN: k={BEST_K}, weights={BEST_WEIGHTS} | Tree: {TREE_MAX_DEPTH}, {TREE_CRITERION} | LR: {LR_CLASS_WEIGHT}")

# Для логістичної регресії прибираємо Poutcome_unknown, бо він дублює Previously_Contacted
# (мультиколінеарність)
lr_drop = ["Poutcome_unknown"]

models = {
    # Baseline: ігнорує ознаки і завжди видає найчастіший клас (0 = "Ні")
    BASELINE_NAME: (DummyClassifier(strategy="most_frequent"), []),
    "k-NN": (Pipeline([("scaler", StandardScaler()),
                       ("knn", KNeighborsClassifier(n_neighbors=BEST_K, weights=BEST_WEIGHTS))]), []),
    "Decision Tree": (DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, criterion=TREE_CRITERION,
                                             random_state=42), []),
    "Logistic Regression": (Pipeline([("scaler", StandardScaler()),
                                      ("lr", LogisticRegression(class_weight=LR_CLASS_WEIGHT,
                                                                max_iter=1000, random_state=42))]),
                            lr_drop),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), []),
    "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=42), []),
}

metrics_rows = []
predictions = pd.DataFrame({"y_true": y_test})

for name, (model, drop_cols) in models.items():
    Xtr = X_train.drop(columns=drop_cols)
    Xte = X_test.drop(columns=drop_cols)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        #baseline ніколи не передбачає Так
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    predictions[name + "_pred"] = y_pred
    predictions[name + "_proba"] = y_proba
    print("Done:", name)

metrics = pd.DataFrame(metrics_rows).round(4)
metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
predictions.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

with open(params_path, "w", encoding="utf-8") as f:
    json.dump({
        "BEST_K": BEST_K,
        "BEST_WEIGHTS": BEST_WEIGHTS,
        "TREE_MAX_DEPTH": TREE_MAX_DEPTH,
        "TREE_CRITERION": TREE_CRITERION,
        "LR_CLASS_WEIGHT": LR_CLASS_WEIGHT,
    }, f, indent=2)

print("\n" + metrics.to_string(index=False))

# БЕЗ "Last Contact Duration" (перевірка витоку даних)
LEAKY_COL = "Last Contact Duration"
no_dur_rows = []
for name, (model, drop_cols) in models.items():
    cols = drop_cols + [LEAKY_COL]
    m = clone(model)  # свіжа копія моделі, щоб перший запуск не зазнав змін
    m.fit(X_train.drop(columns=cols), y_train)

    Xte = X_test.drop(columns=cols)
    y_pred = m.predict(Xte)
    y_proba = m.predict_proba(Xte)[:, 1]

    no_dur_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

no_duration = pd.DataFrame(no_dur_rows).round(4)
no_duration.to_csv(os.path.join(RESULTS_DIR, "metrics_without_duration.csv"), index=False)

print("\nБЕЗ Last Contact Duration:\n" + no_duration.to_string(index=False))

cols_cmp = ["F1-score", "ROC-AUC", "MCC"]
drop_in_quality = metrics.set_index("Model")[cols_cmp] - no_duration.set_index("Model")[cols_cmp]
print("\nПадіння якості після видалення ознаки (з - без):")
print(drop_in_quality.round(4))