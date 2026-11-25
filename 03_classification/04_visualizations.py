import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Підтягуємо збережені параметри моделей з попереднього скрипта
with open(os.path.join(RESULTS_DIR, "params.json"), "r", encoding="utf-8") as f:
    params = json.load(f)

BEST_K = params["BEST_K"]
BEST_WEIGHTS = params["BEST_WEIGHTS"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_CRITERION = params["TREE_CRITERION"]
LR_CLASS_WEIGHT = params["LR_CLASS_WEIGHT"]

# Завантажуємо метрики та передбачення
metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
pred = pd.read_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"))
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_class.csv"))["Subscription"]
model_names = list(metrics["Model"])


# 1. Порівняння моделей (метрики, матриці помилок, ROC та PR-криві)
metrics.set_index("Model").plot(kind="bar", figsize=(12, 6), width=0.8)
plt.ylabel("Значення метрики")
plt.xlabel("Модель")
plt.title("Порівняння моделей на тестовій вибірці (Last Contact Duration - включено)")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "models_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# Матриці помилок для всіх моделей
fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4.5))
for ax, name in zip(axes, model_names):
    cm = confusion_matrix(y_test, pred[name + "_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Ні", "Так"], yticklabels=["Ні", "Так"])
    ax.set_title(name)
    ax.set_xlabel("Передбачено")
    ax.set_ylabel("Фактично")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrices.png"), dpi=300)
plt.close()

# ROC та Precision-Recall криві
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
for name in model_names:
    proba = pred[name + "_proba"]

    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax_pr.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", label="Випадковий класифікатор")
ax_roc.set_xlabel("Частка хибнопозитивних (FPR)")
ax_roc.set_ylabel("Частка істинно позитивних (TPR)")
ax_roc.set_title("ROC-криві")
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)

ax_pr.axhline(y_test.mean(), color="k", linestyle="--", label="Випадковий класифікатор")
ax_pr.set_xlabel("Повнота (Recall)")
ax_pr.set_ylabel("Точність (Precision)")
ax_pr.set_title("Криві Precision-Recall")
ax_pr.legend(loc="upper right")
ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "roc_pr_curves.png"), dpi=300)
plt.close()
print("Частина 1 готова: models_comparison, confusion_matrices, roc_pr_curves")


# 2. Важливість ознак та сигмоїда для логістичної регресії
X_train_lr = X_train.drop(columns=["Poutcome_unknown"])
X_test_lr = X_test.drop(columns=["Poutcome_unknown"])

lr = Pipeline([("scaler", StandardScaler()),
               ("lr", LogisticRegression(class_weight=LR_CLASS_WEIGHT, max_iter=1000, random_state=42))])
lr.fit(X_train_lr, y_train)

coefs = pd.Series(lr.named_steps["lr"].coef_[0], index=X_train_lr.columns)
top15 = coefs.reindex(coefs.abs().sort_values(ascending=False).head(15).index).iloc[::-1]

colors = ["tab:red" if c < 0 else "tab:green" for c in top15.values]
plt.figure(figsize=(9, 7))
plt.barh(top15.index, top15.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Логістична регресія: топ-15 коефіцієнтів\n(зелений — збільшує шанс підписки, червоний — зменшує)")
plt.xlabel("Коефіцієнт (ознаки стандартизовані)")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "logistic_regression_insights.png"), dpi=300)
plt.close()

z_scores = lr.decision_function(X_test_lr)
probs = lr.predict_proba(X_test_lr)[:, 1]
order = np.argsort(z_scores)

plt.figure(figsize=(9, 5))
plt.plot(z_scores[order], probs[order], color="tab:green", linewidth=2.5, label="Сигмоїда (ймовірність моделі)")
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Поріг рішення (0.5)")

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(y_test), size=400, replace=False)
for cls, color, lbl in [(0, "tab:red", "Відмова (0)"), (1, "tab:blue", "Підписка (1)")]:
    m = y_test.values[sample_idx] == cls
    plt.scatter(z_scores[sample_idx][m], y_test.values[sample_idx][m],
                c=color, alpha=0.5, s=25, label=lbl)

plt.title("Логістична регресія: S-крива ймовірності підписки на тестових даних")
plt.xlabel("Лінійна комбінація ознак (z)")
plt.ylabel("Ймовірність підписки P(Так)")
plt.legend(loc="center right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "logistic_sigmoid_curve.png"), dpi=300)
plt.close()
print("Частина 2 готова: logistic_regression_insights, logistic_sigmoid_curve")


# 3. Підрахунок помилок (FN та FP) + графік впливу Last Contact Duration
rows = []
for name in model_names:
    tn, fp, fn, tp = confusion_matrix(y_test, pred[name + "_pred"]).ravel()
    rows.append({
        "Model": name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Missed subscribers (FN/(FN+TP))": fn / (fn + tp),
        "False alarms (FP/(FP+TN))": fp / (fp + tn),
    })

errors = pd.DataFrame(rows).round(4)
errors.to_csv(os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False)

no_dur_path = os.path.join(RESULTS_DIR, "metrics_without_duration.csv")
if os.path.exists(no_dur_path):
    no_dur = pd.read_csv(no_dur_path)
    # Фільтруємо Baseline, щоб на графіку були тільки робочі моделі
    m_plot = metrics[metrics["Model"] != "Baseline (always No)"].reset_index(drop=True)
    nd_plot = no_dur[no_dur["Model"] != "Baseline (always No)"].reset_index(drop=True)

    x = np.arange(len(m_plot))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "MCC", "Вплив тривалості дзвінка на MCC"),
        (axes[1], "ROC-AUC", "Вплив тривалості дзвінка на ROC-AUC"),
    ]:
        b1 = ax.bar(x - width / 2, m_plot[metric], width, label="З Last Contact Duration", color="tab:blue")
        b2 = ax.bar(x + width / 2, nd_plot[metric], width, label="Без Last Contact Duration", color="tab:orange")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(m_plot["Model"], rotation=20)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "duration_leakage_impact.png"), dpi=300)
    plt.close()
    print("Частина 3 готова: error_analysis.csv, duration_leakage_impact.png")
else:
    print("Частина 3 готова: error_analysis.csv")

print(errors.to_string(index=False))