#  Задача 1. Регресія: підбір гіперпараметрів
#  Цільова змінна: Balance (баланс рахунку клієнта, євро).
#  Параметри підбираються 5-блочною крос-валідацією тільки на train,
#  тестова вибірка тут не використовується.
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reg.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reg.csv"))["Balance"]
print("Навчальна вибірка:", X_train.shape)

# однаковий поділ на блоки для всіх моделей
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def cv_r2(model):
    # повертає середній R^2 на train-блоках і на валідаційних блоках
    res = cross_validate(model, X_train, y_train, cv=cv, scoring="r2",
                         return_train_score=True, n_jobs=-1)
    return [res["train_score"].mean(), res["test_score"].mean()]


# k-NN: кількість сусідів
print("\nk-NN: підбір кількості сусідів k")
k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]
knn_rows = []
best_k = 0
best_knn_r2 = -100
for k in k_values:
    knn = Pipeline([("scaler", StandardScaler()),
                    ("knn", KNeighborsRegressor(n_neighbors=k))])
    scores = cv_r2(knn)
    knn_rows.append([k, scores[0], scores[1]])
    print("  k =", k, " R2 train =", round(scores[0], 4),
          " R2 CV =", round(scores[1], 4))
    if scores[1] > best_knn_r2:
        best_knn_r2 = scores[1]
        best_k = k
knn_table = pd.DataFrame(knn_rows, columns=["k", "R2_train", "R2_cv"])
knn_table.to_csv(os.path.join(RESULTS_DIR, "tuning_knn.csv"), index=False)
print("Найкраще k =", best_k, "R2 CV =", round(best_knn_r2, 4))


# Дерево рішень: глибина та мінімальна кількість об'єктів у листі
print("\nDecision Tree: підбір max_depth та min_samples_leaf")
depth_values = [2, 3, 4, 5, 6, 8, 10, 12, 15]
leaf_values = [1, 20, 50, 100]
tree_rows = []
best_depth = 0
best_leaf = 0
best_tree_r2 = -100
for leaf in leaf_values:
    for depth in depth_values:
        tree = DecisionTreeRegressor(max_depth=depth,
                                     min_samples_leaf=leaf,
                                     random_state=42)
        scores = cv_r2(tree)
        tree_rows.append([depth, leaf, scores[0], scores[1]])
        if scores[1] > best_tree_r2:
            best_tree_r2 = scores[1]
            best_depth = depth
            best_leaf = leaf
tree_table = pd.DataFrame(tree_rows, columns=["max_depth", "min_samples_leaf",
                                              "R2_train", "R2_cv"])
tree_table.to_csv(os.path.join(RESULTS_DIR, "tuning_tree.csv"), index=False)
print(tree_table.round(4).to_string(index=False))
print("Найкраще дерево: max_depth =", best_depth,
      "min_samples_leaf =", best_leaf, "R2 CV =", round(best_tree_r2, 4))


# Random Forest: мінімальна кількість об'єктів у листі
# (кількість дерев фіксована, 100 дерев достатньо для порівняння)
print("\nRandom Forest: підбір min_samples_leaf")
rf_leaf_values = [1, 5, 10, 20, 50, 100]
rf_rows = []
best_rf_leaf = 0
best_rf_r2 = -100
for leaf in rf_leaf_values:
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf,
                               random_state=42, n_jobs=-1)
    scores = cv_r2(rf)
    rf_rows.append([leaf, scores[0], scores[1]])
    print("  min_samples_leaf =", leaf, " R2 train =", round(scores[0], 4),
          " R2 CV =", round(scores[1], 4))
    if scores[1] > best_rf_r2:
        best_rf_r2 = scores[1]
        best_rf_leaf = leaf
rf_table = pd.DataFrame(rf_rows, columns=["min_samples_leaf", "R2_train", "R2_cv"])
rf_table.to_csv(os.path.join(RESULTS_DIR, "tuning_rf.csv"), index=False)
print("Найкраще min_samples_leaf =", best_rf_leaf, "R2 CV =", round(best_rf_r2, 4))

# зберігаємо параметри для наступного скрипта
with open(os.path.join(RESULTS_DIR, "params.json"), "w", encoding="utf-8") as f:
    json.dump({"BEST_K": best_k,
               "TREE_MAX_DEPTH": best_depth,
               "TREE_MIN_LEAF": best_leaf,
               "RF_MIN_LEAF": best_rf_leaf}, f, indent=2)


# Графіки впливу параметрів
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(knn_table["k"], knn_table["R2_train"], "o--", color="gray",
        label="train")
ax.plot(knn_table["k"], knn_table["R2_cv"], "o-", color="tab:blue",
        label="крос-валідація")
ax.axvline(best_k, color="red", linestyle=":", label="обране k = " + str(best_k))
ax.set_title("k-NN: вплив кількості сусідів")
ax.set_xlabel("k")
ax.set_ylabel("R²")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
for i in range(len(leaf_values)):
    part = tree_table[tree_table["min_samples_leaf"] == leaf_values[i]]
    ax.plot(part["max_depth"], part["R2_cv"], "o-", color=colors[i],
            label="min_samples_leaf = " + str(leaf_values[i]))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylim(-0.3, 0.1)
ax.set_title("Дерево рішень: R² на крос-валідації")
ax.set_xlabel("max_depth")
ax.set_ylabel("R²")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(rf_table["min_samples_leaf"], rf_table["R2_train"], "o--", color="gray",
        label="train")
ax.plot(rf_table["min_samples_leaf"], rf_table["R2_cv"], "o-", color="tab:green",
        label="крос-валідація")
ax.set_xscale("log")
ax.set_title("Random Forest: вплив min_samples_leaf")
ax.set_xlabel("min_samples_leaf")
ax.set_ylabel("R²")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "tuning_impact.png"), dpi=150)
plt.close()
print("\nГрафік збережено: images/tuning_impact.png")
