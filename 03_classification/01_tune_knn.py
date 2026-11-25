import os
import pandas as pd
import matplotlib
import json
matplotlib.use("Agg") # no GUI window
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_class.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_class.csv"))["Subscription"]

# крос-валідація на 5 фолдів
# StratifiedKFold використовується бо сильний дисбаланс класів (88%/12%)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41]  # непарні числа, щоб уникнути нічиєї
weights_options = ["uniform", "distance"]           # рівні голоси vs вага за відстанню

rows = []
for w in weights_options:
    for k in k_values:
        # Pipeline спочатку масштабування, потім сам KNN
        # StandardScaler приюирає перебивка відстаней (бо в ознаках є вел.числа)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=w)),
        ])

        # для того щоб модель не перенавчалась на трейн наборі, 
        # по черзі вчимо на 4 і перевіряємо на 1
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="matthews_corrcoef", n_jobs=-1)
        rows.append({
            "weights": w,
            "k": k,
            "MCC (mean)": scores.mean(),
            "MCC (std)": scores.std()})

results = pd.DataFrame(rows)
# пошук комбінації з максимальним середнім MCC
best_idx = results["MCC (mean)"].idxmax()
best_row = results.loc[best_idx]
best_k = int(best_row["k"])
best_w = best_row["weights"]

print(f"\n{'weights':<9} | {'k':>3} | {'MCC mean':>8} | {'MCC std':>8}")
print("-" * 38)
for idx, row in results.iterrows():
    flag = " <-- best" if idx == best_idx else ""
    print(f"{row['weights']:<9} | {int(row['k']):>3} | {row['MCC (mean)']:>8.4f} | {row['MCC (std)']:>8.4f}{flag}")
print("-" * 38)

print(f"\nНайкращий варіант: k = {best_k}, weights = '{best_w}'")
print(f"MCC: {best_row['MCC (mean)']:.4f} (std: {best_row['MCC (std)']:.4f})")
print(f"Значення для 03_train_evaluate_models.py: BEST_K = {best_k}, BEST_WEIGHTS = '{best_w}'\n")


RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
params_path = os.path.join(RESULTS_DIR, "params.json")

params = json.load(open(params_path, encoding="utf-8")) if os.path.exists(params_path) else {}
params.update({"BEST_K": best_k, "BEST_WEIGHTS": best_w})
with open(params_path, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

# графік залежності метрики MCC від k для обох типів ваг
plt.figure(figsize=(8, 5))
for w, style in [("uniform", "bo--"), ("distance", "go-")]:
    subset = results[results["weights"] == w]
    plt.plot(subset["k"], subset["MCC (mean)"], style, label=f"weights = '{w}'")

plt.title("Залежність якості моделі (MCC) від k та типу ваг")
plt.xlabel("Кількість сусідів (k)")
plt.ylabel("Метрика MCC")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(IMAGES_DIR, "knn_k_vs_mcc.png"), dpi=300)
plt.close()