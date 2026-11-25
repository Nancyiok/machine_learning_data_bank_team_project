#  Задача 1. Регресія: графіки
#  1) розподіл цільової змінної Balance;
#  2) порівняння моделей за MAE, RMSE, R^2;
#  3) фактичні та прогнозовані значення для кожної моделі;
#  4) залишки найкращої моделі;
#  5) важливість ознак (Linear Regression і Random Forest);
#  6) середній фактичний і прогнозований баланс за віковими групами.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_FILE = os.path.join(PROJECT_ROOT, "raw_data", "Bank Marketing.csv")
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reg.csv"))["Balance"]
metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
pred = pd.read_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"))
coefs = pd.read_csv(os.path.join(RESULTS_DIR, "linear_coefficients.csv"))
importance = pd.read_csv(os.path.join(RESULTS_DIR, "rf_permutation_importance.csv"))
y_test = pred["y_true"]

model_names = ["Linear Regression", "k-NN", "Decision Tree", "Random Forest"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# 1. Розподіл цільової змінної
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
# ширина стовпчика 250 євро
ax.hist(y_train, bins=np.arange(-3000, 15001, 250), color="steelblue")
ax.axvline(y_train.mean(), color="red", linestyle="--",
           label="середнє = " + str(round(y_train.mean())))
ax.axvline(y_train.median(), color="green", linestyle="--",
           label="медіана = " + str(round(y_train.median())))
ax.set_xlim(-3000, 15000)
ax.set_title("Розподіл Balance (train), обрізано до 15 000 €")
ax.set_xlabel("Баланс, євро")
ax.set_ylabel("Кількість клієнтів")
ax.legend()

ax = axes[1]
log_y = np.sign(y_train) * np.log1p(np.abs(y_train))
ax.hist(log_y, bins=100, color="darkorange")
ax.set_title("Розподіл sign(y)·ln(1 + |y|)")
ax.set_xlabel("Перетворений баланс")
ax.set_ylabel("Кількість клієнтів")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "target_distribution.png"), dpi=150)
plt.close()


# 2. Порівняння моделей
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
metric_list = ["MAE", "RMSE", "R2"]
titles = ["MAE, євро (менше краще)", "RMSE, євро (менше краще)",
          "R² (більше краще)"]
bar_colors = ["gray"] + colors
for j in range(3):
    ax = axes[j]
    values = metrics[metric_list[j]]
    ax.bar(metrics["Model"], values, color=bar_colors)
    for i in range(len(values)):
        value = round(values[i], 3)
        if value == 0:
            value = 0.0  # щоб не друкувалося -0.0
        ax.text(i, values[i], str(value), ha="center",
                va="bottom", fontsize=9)
    ax.set_title(titles[j])
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylim(1300, 1600)
axes[1].set_ylim(2900, 3200)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "models_comparison.png"), dpi=150)
plt.close()


# 3. Фактичні та прогнозовані значення
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for i in range(4):
    ax = axes[i // 2][i % 2]
    name = model_names[i]
    r2 = metrics.loc[metrics["Model"] == name, "R2"].values[0]
    ax.scatter(y_test, pred[name], s=6, alpha=0.3, color=colors[i])
    ax.plot([-3000, 20000], [-3000, 20000], "k--", label="y = x")
    ax.axhline(y_train.mean(), color="gray", linestyle=":",
               label="середнє train")
    ax.set_xlim(-3000, 20000)
    ax.set_ylim(-3000, 20000)
    ax.set_title(name + ", R² = " + str(round(r2, 3)))
    ax.set_xlabel("Фактичний баланс, євро")
    ax.set_ylabel("Прогнозований баланс, євро")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()


# 4. Залишки Random Forest
best = "Random Forest"
residuals = y_test - pred[best]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(pred[best], residuals, s=6, alpha=0.3, color="tab:red")
ax.axhline(0, color="black")
ax.set_ylim(-5000, 20000)
ax.set_title("Залишки " + best)
ax.set_xlabel("Прогноз, євро")
ax.set_ylabel("Фактичне - прогноз, євро")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(residuals, bins=np.arange(-5000, 15001, 250), color="tab:red")
ax.set_xlim(-5000, 15000)
ax.set_title("Розподіл залишків " + best)
ax.set_xlabel("Залишок, євро")
ax.set_ylabel("Кількість клієнтів")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "residuals.png"), dpi=150)
plt.close()


# 5. Важливість ознак
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
# 6 найменших і 6 найбільших коефіцієнтів
top_coefs = pd.concat([coefs.head(6), coefs.tail(6)])
bar_col = []
for value in top_coefs["Coef"]:
    if value < 0:
        bar_col.append("tab:red")
    else:
        bar_col.append("tab:green")
ax.barh(top_coefs["Feature"], top_coefs["Coef"], color=bar_col)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Linear Regression: найбільші коефіцієнти, євро")
ax.grid(True, axis="x", alpha=0.3)

ax = axes[1]
top_imp = importance.head(12).iloc[::-1]
ax.barh(top_imp["Feature"], top_imp["Importance"], xerr=top_imp["Std"],
        color="tab:red")
ax.set_title("Random Forest: падіння R² при перемішуванні ознаки")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150)
plt.close()


# 6. Середній баланс за віковими групами
# вік беремо з сирих даних, бо в X_test він масштабований
raw = pd.read_csv(RAW_FILE)
# той самий поділ, що й у data_preprocessing.py (random_state=42)
raw_train, raw_test = train_test_split(raw, test_size=0.2, random_state=42)
if not (raw_test["Balance (euros)"].values == y_test.values).all():
    print("Увага: поділ не збігається з processed_data")
age = raw_test["Age"].values
groups = pd.cut(age, bins=[17, 30, 40, 50, 60, 100],
                labels=["18-30", "31-40", "41-50", "51-60", "60+"])
by_age = pd.DataFrame({"group": groups, "fact": y_test.values,
                       "lr": pred["Linear Regression"].values,
                       "rf": pred["Random Forest"].values})
by_age = by_age.groupby("group", observed=True).mean()
by_age.to_csv(os.path.join(RESULTS_DIR, "balance_by_age.csv"))
print("Середній баланс за віковими групами (test):")
print(by_age.round(1).to_string())

x = np.arange(len(by_age))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.27, by_age["fact"], width=0.27, color="gray", label="фактичний")
plt.bar(x, by_age["lr"], width=0.27, color="tab:blue",
        label="Linear Regression")
plt.bar(x + 0.27, by_age["rf"], width=0.27, color="tab:red",
        label="Random Forest")
plt.xticks(x, by_age.index)
plt.title("Середній баланс за віковими групами, тестова вибірка")
plt.xlabel("Вік")
plt.ylabel("Баланс, євро")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "balance_by_age.png"), dpi=150)
plt.close()

print("Графіки збережено в папку images")
