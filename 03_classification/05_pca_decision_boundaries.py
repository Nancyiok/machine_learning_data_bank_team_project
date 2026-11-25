import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # без GUI-вікна
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Підтягуємо збережені параметри моделей з попередніх скриптів
with open(os.path.join(RESULTS_DIR, "params.json"), "r", encoding="utf-8") as f:
    params = json.load(f)

BEST_K = params["BEST_K"]
BEST_WEIGHTS = params["BEST_WEIGHTS"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_CRITERION = params["TREE_CRITERION"]
LR_CLASS_WEIGHT = params["LR_CLASS_WEIGHT"]

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"].astype(int)
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_class.csv"))["Subscription"].astype(int)

# 7 ознак, на яких робилась кластеризація (спільні для кластеризації та класифікації)
features_7 = [
    "Age", "Balance (euros)", "Education", "Housing Loan_yes",
    "Personal Loan_yes", "Marital Status_married", "Marital Status_single",
]
# Усі ознаки, відомі ДО дзвінка (без Last Contact Duration, щоб уникнути витоку даних)
cols_no_dur = [c for c in X_train.columns if c != "Last Contact Duration"]


def get_models():
    return {
        f"k-NN (k={BEST_K}, {BEST_WEIGHTS})": KNeighborsClassifier(
            n_neighbors=BEST_K, weights=BEST_WEIGHTS),
        f"Decision Tree ({TREE_CRITERION}, depth={TREE_MAX_DEPTH})": DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH, criterion=TREE_CRITERION, random_state=42),
        f"Logistic Regression ({LR_CLASS_WEIGHT})": LogisticRegression(
            class_weight=LR_CLASS_WEIGHT, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

def solid_legend(ax, **kwargs):
    leg = ax.legend(**kwargs)
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    for lh in handles:
        lh.set_alpha(1.0)
        if hasattr(lh, "set_sizes"):
            lh.set_sizes([30])

def fix_pca_sign(pca, *projections):
    for i in range(2):
        if pca.components_[i].sum() < 0:
            pca.components_[i] *= -1
            for Z in projections:
                Z[:, i] *= -1


# 1. Порівняння кластеризації та класифікації (7 спільних ознак)
X_all_7 = pd.concat([X_train[features_7], X_test[features_7]], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True).to_numpy()

# Обрізаємо викиди в Balance по 95-му перцентилю, щоб вони не домінували в PCA
limit_all = X_all_7["Balance (euros)"].quantile(0.95)
X_all_7["Balance (euros)"] = X_all_7["Balance (euros)"].clip(upper=limit_all)

pca_all = PCA(n_components=2, random_state=42)
Z_all_7 = pca_all.fit_transform(X_all_7)
fix_pca_sign(pca_all, Z_all_7)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_all = kmeans.fit_predict(X_all_7)
centers_2d = pca_all.transform(kmeans.cluster_centers_)

# Класифікатор вчимо на train, передбачаємо на test (ознаки до дзвінка)
classifier = HistGradientBoostingClassifier(random_state=42)
classifier.fit(X_train[cols_no_dur], y_train)
y_pred_test = classifier.predict(X_test[cols_no_dur])
Z_test_all = Z_all_7[len(X_train):]

# Для читабельності малюємо випадкову підвибірку (однакову для всіх панелей)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_all_7), size=min(6000, len(X_all_7)), replace=False)
test_sample = rng.choice(len(Z_test_all), size=min(3000, len(Z_test_all)), replace=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
class_colors = {0: "#1f77b4", 1: "#d62728"}

for c in range(4):
    idx = sample_idx[clusters_all[sample_idx] == c]
    axes[0].scatter(Z_all_7[idx, 0], Z_all_7[idx, 1], s=8, alpha=0.5,
                    color=cluster_colors[c], label=f"Кластер {c}")
axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], s=180, marker="X",
                color="black", label="Центри")
axes[0].set_title("K-Means: 4 кластери")

for c, label in [(0, "Не відкрили"), (1, "Відкрили")]:
    idx = sample_idx[y_all[sample_idx] == c]
    axes[1].scatter(Z_all_7[idx, 0], Z_all_7[idx, 1], s=8, alpha=0.5,
                    color=class_colors[c], label=label)
axes[1].set_title("Фактичне відкриття депозиту")

for c, label in [(0, "Прогноз: ні"), (1, "Прогноз: так")]:
    mask = y_pred_test[test_sample] == c
    axes[2].scatter(Z_test_all[test_sample][mask, 0], Z_test_all[test_sample][mask, 1],
                    s=8, alpha=0.5, color=class_colors[c], label=label)
axes[2].set_title(f"Прогноз на тестовій вибірці ({len(cols_no_dur)} ознак)")

pc1_a, pc2_a = 100 * pca_all.explained_variance_ratio_
for ax in axes:
    ax.set_xlabel(f"Головна компонента 1 ({pc1_a:.1f}%)")
    ax.set_ylabel(f"Головна компонента 2 ({pc2_a:.1f}%)")
    ax.grid(True, alpha=0.3)
    solid_legend(ax, loc="upper left")

plt.suptitle("Порівняння кластеризації та класифікації через PCA (7 спільних ознак)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pca_clusters_vs_classification.png"), dpi=200)
plt.close()


# 2. Межі рішень у просторі 7 ознак кластеризації
X_tr_7 = X_train[features_7].copy()
X_te_7 = X_test[features_7].copy()

# Поріг обрізання беремо ТІЛЬКИ з train, щоб test не впливав на підготовку
bal_lim = X_tr_7["Balance (euros)"].quantile(0.95)
X_tr_7["Balance (euros)"] = X_tr_7["Balance (euros)"].clip(upper=bal_lim)
X_te_7["Balance (euros)"] = X_te_7["Balance (euros)"].clip(upper=bal_lim)

pca_7 = PCA(n_components=2, random_state=42)
Z_tr_7 = pca_7.fit_transform(X_tr_7)
Z_te_7 = pca_7.transform(X_te_7)
fix_pca_sign(pca_7, Z_tr_7, Z_te_7)

pc1_7, pc2_7 = 100 * pca_7.explained_variance_ratio_

# Межі осей з перцентилів (замість жорстко заданих чисел)
x_min, x_max = np.percentile(Z_tr_7[:, 0], [1, 99])
y_min, y_max = np.percentile(Z_tr_7[:, 1], [1, 99])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_7 = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (name, model) in zip(axes, get_models().items()):
    model.fit(Z_tr_7, y_train)
    zz = model.predict(grid_7).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.28, levels=[-0.5, 0.5, 1.5], colors=["#1f77b4", "#d62728"])
    for cls, color, lbl, alpha, size in [(0, "#1f77b4", "Ні (0)", 0.30, 5),
                                         (1, "#d62728", "Так (1)", 0.45, 6)]:
        mask = y_test.values == cls
        ax.scatter(Z_te_7[mask, 0], Z_te_7[mask, 1], c=color, s=size,
                   alpha=alpha, linewidth=0, label=lbl)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Межа рішень: {name}")
    ax.set_xlabel(f"PC1 ({pc1_7:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_7:.1f}%)")
    ax.grid(alpha=0.25)
    solid_legend(ax, title="Підписка (test)", loc="upper left")

axes[-1].axis("off")  # 5 моделей на сітці 2x3: остання клітинка порожня
fig.suptitle(f"Межі рішень у просторі 7 ознак кластеризації (test: {len(Z_te_7)} клієнтів)",
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_boundaries_7features.png"), dpi=200)
plt.close()


# Межі рішень на всіх ознаках до дзвінка (окремий PCA для класифікації)
# ознаки різних масштабів, тому спочатку StandardScaler, потім PCA
scaler_full = StandardScaler()
X_tr_scaled = scaler_full.fit_transform(X_train[cols_no_dur])
X_te_scaled = scaler_full.transform(X_test[cols_no_dur])

pca_full = PCA(n_components=2, random_state=42)
Z_tr_full = pca_full.fit_transform(X_tr_scaled)
Z_te_full = pca_full.transform(X_te_scaled)

pc1_f, pc2_f = 100 * pca_full.explained_variance_ratio_

# ознаки що найбільше формують PC1 та PC2
loadings = pd.DataFrame(pca_full.components_.T, index=cols_no_dur, columns=["PC1", "PC2"])
print("Топ-5 ознак для PC1:\n", loadings["PC1"].abs().sort_values(ascending=False).head(5).round(3))
print("Топ-5 ознак для PC2:\n", loadings["PC2"].abs().sort_values(ascending=False).head(5).round(3))
x_min, x_max = np.percentile(Z_tr_full[:, 0], [1, 99])
y_min, y_max = np.percentile(Z_tr_full[:, 1], [1, 99])
xx_f, yy_f = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_f = np.c_[xx_f.ravel(), yy_f.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (name, model) in zip(axes, get_models().items()):
    model.fit(Z_tr_full, y_train)
    zz = model.predict(grid_f).reshape(xx_f.shape)
    ax.contourf(xx_f, yy_f, zz, alpha=0.28, levels=[-0.5, 0.5, 1.5], colors=["#1f77b4", "#d62728"])
    for cls, color, lbl, alpha, size in [(0, "#1f77b4", "Ні (0)", 0.25, 5),
                                         (1, "#d62728", "Так (1)", 0.55, 7)]:
        mask = y_test.values == cls
        ax.scatter(Z_te_full[mask, 0], Z_te_full[mask, 1], c=color, s=size,
                   alpha=alpha, linewidth=0, label=lbl)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Межа рішень: {name}")
    ax.set_xlabel(f"PC1 ({pc1_f:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_f:.1f}%)")
    ax.grid(alpha=0.25)
    solid_legend(ax, title="Підписка (test)", loc="upper right")

axes[-1].axis("off")
fig.suptitle(f"Межі рішень класифікаторів на всіх {len(cols_no_dur)} ознаках до дзвінка "
             f"(test: {len(Z_te_full)} клієнтів)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_boundaries_all_features.png"), dpi=200)
plt.close()