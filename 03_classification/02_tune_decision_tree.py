import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # без GUI-вікна
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, GridSearchCV

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "max_depth": [3, 5, 8, 12, None],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                    scoring="matthews_corrcoef", cv=cv,
                    return_train_score=True, n_jobs=-1)
grid.fit(X_train, y_train)

table = pd.DataFrame({
    "criterion": grid.cv_results_["param_criterion"],
    "max_depth": [str(d) for d in grid.cv_results_["param_max_depth"]],
    "Train MCC": grid.cv_results_["mean_train_score"],
    "Val MCC": grid.cv_results_["mean_test_score"],
    "MCC (std)": grid.cv_results_["std_test_score"],
})

best_idx = grid.best_index_
best_row = table.loc[best_idx]
best_depth = grid.best_params_["max_depth"]
best_crit = grid.best_params_["criterion"]

print(f"\n{'criterion':<9} | {'max_depth':<9} | {'Train MCC':>9} | {'Val MCC':>8} | {'MCC (std)':>9}")
print("-" * 56)
for idx, row in table.iterrows():
    mark = "  <-- best" if idx == best_idx else ""
    print(f"{row['criterion']:<9} | {row['max_depth']:<9} | {row['Train MCC']:>9.4f} | "
          f"{row['Val MCC']:>8.4f} | {row['MCC (std)']:>9.4f}{mark}")
print("-" * 56)

print(f"\nНайкращі параметри: max_depth = {best_depth}, criterion = '{best_crit}'")
print(f"Val MCC: {best_row['Val MCC']:.4f} (std: {best_row['MCC (std)']:.4f}) | Train MCC: {best_row['Train MCC']:.4f}")
print(f"Для 03_train_evaluate_models.py: TREE_MAX_DEPTH = {best_depth}, TREE_CRITERION = '{best_crit}'\n")

params_path = os.path.join(RESULTS_DIR, "params.json")

params = json.load(open(params_path, encoding="utf-8")) if os.path.exists(params_path) else {}
params.update({"TREE_MAX_DEPTH": best_depth, "TREE_CRITERION": best_crit})
with open(params_path, "w", encoding="utf-8") as f: json.dump(params, f, indent=2)

# MCC vs max_depth для критеріїв Джині та Ентропії
depth_labels = [str(d) for d in param_grid["max_depth"]]
x_pos = range(len(depth_labels))

plt.figure(figsize=(8, 5))
for crit, style in [("gini", "bo--"), ("entropy", "go-")]:
    subset = table[table["criterion"] == crit]
    plt.plot(x_pos, subset["Val MCC"], style, label=f"criterion = '{crit}'")

plt.title("Залежність якості моделі (MCC) від max_depth та критерію")
plt.xlabel("Максимальна глибина дерева (max_depth)")
plt.ylabel("Метрика MCC")
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(IMAGES_DIR, "tree_depth_vs_mcc.png"), dpi=300)
plt.close()

# Для найкращої моделі відбираємо топ-15 найважливіших ознак
tree = grid.best_estimator_
importances = pd.Series(tree.feature_importances_, index=X_train.columns)
top15 = importances.sort_values(ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={"width_ratios": [1, 2]})

axes[0].barh(top15.index[::-1], top15.values[::-1])
axes[0].set_title("Top-15 feature importances")
axes[0].set_xlabel("Importance")

plot_tree(tree, max_depth=3, feature_names=X_train.columns,
          class_names=["No", "Yes"], filled=True, fontsize=7, ax=axes[1])
axes[1].set_title("Decision Tree (top 3 levels)")

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_tree_insights.png"), dpi=300)
plt.close()