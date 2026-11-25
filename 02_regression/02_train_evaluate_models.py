#  Задача 1. Регресія: навчання та оцінювання моделей
#  Моделі: Linear Regression, k-NN, Decision Tree, Random Forest.
#  Baseline: DummyRegressor, який завжди прогнозує середній баланс train.
#  Метрики на тестовій вибірці: MAE, RMSE, R^2.
import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reg.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reg.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reg.csv"))["Balance"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_reg.csv"))["Balance"]

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Середній баланс train:", round(y_train.mean(), 1),
      " медіана:", y_train.median())

with open(os.path.join(RESULTS_DIR, "params.json"), encoding="utf-8") as f:
    params = json.load(f)
BEST_K = params["BEST_K"]
TREE_MAX_DEPTH = params["TREE_MAX_DEPTH"]
TREE_MIN_LEAF = params["TREE_MIN_LEAF"]
RF_MIN_LEAF = params["RF_MIN_LEAF"]
print("k-NN: k =", BEST_K, "| Tree: max_depth =", TREE_MAX_DEPTH,
      "min_samples_leaf =", TREE_MIN_LEAF, "| RF: min_samples_leaf =", RF_MIN_LEAF)

model_names = ["Baseline (mean)", "Linear Regression", "k-NN",
               "Decision Tree", "Random Forest"]
models = [
    DummyRegressor(strategy="mean"),
    LinearRegression(),
    Pipeline([("scaler", StandardScaler()),
              ("knn", KNeighborsRegressor(n_neighbors=BEST_K))]),
    DecisionTreeRegressor(max_depth=TREE_MAX_DEPTH,
                          min_samples_leaf=TREE_MIN_LEAF, random_state=42),
    RandomForestRegressor(n_estimators=200, min_samples_leaf=RF_MIN_LEAF,
                          random_state=42, n_jobs=-1),
]


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return [mae, rmse, r2]


rows = []
predictions = pd.DataFrame({"y_true": y_test})
# для лінійної регресії прибираємо Poutcome_unknown: він повністю дублює
# Previously_Contacted (кореляція -1), і коефіцієнти стають безглуздими.
# Так само зроблено в логістичній регресії в задачі класифікації
lr_cols = []
for col in X_train.columns:
    if col != "Poutcome_unknown":
        lr_cols.append(col)

for i in range(len(models)):
    model = models[i]
    if model_names[i] == "Linear Regression":
        cols = lr_cols
    else:
        cols = list(X_train.columns)
    model.fit(X_train[cols], y_train)
    pred_train = model.predict(X_train[cols])
    pred_test = model.predict(X_test[cols])

    m_train = get_metrics(y_train, pred_train)
    m_test = get_metrics(y_test, pred_test)
    # R^2 на train потрібен, щоб побачити перенавчання
    rows.append([model_names[i], m_test[0], m_test[1], m_test[2], m_train[2]])
    predictions[model_names[i]] = pred_test
    print("Готово:", model_names[i])

metrics = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2", "R2_train"])
metrics = metrics.round(4)
metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
predictions.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
print("\nМетрики на тестовій вибірці:")
print(metrics.to_string(index=False))

# найкраща модель за RMSE на тесті (baseline не враховуємо)
best_name = ""
best_rmse = 10 ** 9
for i in range(1, len(rows)):
    if rows[i][2] < best_rmse:
        best_rmse = rows[i][2]
        best_name = rows[i][0]
print("\nНайкраща модель за RMSE:", best_name)


# Рівняння лінійної регресії: коефіцієнти
lin_reg = models[1]
coefs = pd.DataFrame({"Feature": lr_cols, "Coef": lin_reg.coef_})
coefs = coefs.sort_values("Coef")
coefs.to_csv(os.path.join(RESULTS_DIR, "linear_coefficients.csv"), index=False)
print("\nВільний член b0 =", round(lin_reg.intercept_, 2))
print("Найбільші від'ємні коефіцієнти:")
print(coefs.head(5).round(2).to_string(index=False))
print("Найбільші додатні коефіцієнти:")
print(coefs.tail(5).round(2).to_string(index=False))


# Важливість ознак для Random Forest (permutation importance на тесті)
rf = models[4]
perm = permutation_importance(rf, X_test, y_test, scoring="r2",
                              n_repeats=5, random_state=42, n_jobs=-1)
importance = pd.DataFrame({"Feature": X_test.columns,
                           "Importance": perm.importances_mean,
                           "Std": perm.importances_std})
importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(os.path.join(RESULTS_DIR, "rf_permutation_importance.csv"),
                  index=False)
print("\nRandom Forest: падіння R^2 при перемішуванні ознаки (топ-10)")
print(importance.head(10).round(4).to_string(index=False))


# Додатковий експеримент: логарифм цільової змінної
# Баланс має довгий хвіст, тому пробуємо навчати моделі на
# z = sign(y) * ln(1 + |y|) і повертати прогноз назад у євро
def to_log(y):
    return np.sign(y) * np.log1p(np.abs(y))


def from_log(z):
    return np.sign(z) * np.expm1(np.abs(z))


log_rows = []
log_names = ["Linear Regression", "Random Forest"]
log_models = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=200, min_samples_leaf=RF_MIN_LEAF,
                          random_state=42, n_jobs=-1),
]
for i in range(len(log_models)):
    if log_names[i] == "Linear Regression":
        cols = lr_cols
    else:
        cols = list(X_train.columns)
    log_models[i].fit(X_train[cols], to_log(y_train))
    pred = from_log(log_models[i].predict(X_test[cols]))
    m = get_metrics(y_test, pred)
    med_ae = np.median(np.abs(y_test - pred))
    log_rows.append([log_names[i] + " (log y)", m[0], m[1], m[2], med_ae])

# для порівняння ті самі моделі без перетворення і медіана
for name in log_names:
    pred = predictions[name]
    m = get_metrics(y_test, pred)
    med_ae = np.median(np.abs(y_test - pred))
    log_rows.append([name, m[0], m[1], m[2], med_ae])
pred = np.full(len(y_test), y_train.median())
m = get_metrics(y_test, pred)
log_rows.append(["Baseline (median)", m[0], m[1], m[2],
                 np.median(np.abs(y_test - pred))])

log_table = pd.DataFrame(log_rows, columns=["Model", "MAE", "RMSE", "R2", "MedAE"])
log_table = log_table.round(4)
log_table.to_csv(os.path.join(RESULTS_DIR, "metrics_log_target.csv"), index=False)
print("\nВплив логарифмування цільової змінної:")
print(log_table.to_string(index=False))


# Перевірка надійності висновків
# 1) бутстреп тестової вибірки: довірчі інтервали R^2
#    і різниці R^2 між Random Forest та іншими моделями
print("\nБутстреп (1000 повторів): 95 % інтервали R^2 на тесті")
rng = np.random.default_rng(42)
n_test = len(y_test)
y_arr = y_test.values
boot_names = ["Linear Regression", "k-NN", "Decision Tree", "Random Forest"]
boot_r2 = []
for name in boot_names:
    boot_r2.append([])
boot_diff = [[], [], []]
for b in range(1000):
    idx = rng.integers(0, n_test, n_test)
    values = []
    for name in boot_names:
        values.append(r2_score(y_arr[idx], predictions[name].values[idx]))
    for i in range(4):
        boot_r2[i].append(values[i])
    for i in range(3):
        boot_diff[i].append(values[3] - values[i])

boot_rows = []
for i in range(4):
    low = np.percentile(boot_r2[i], 2.5)
    high = np.percentile(boot_r2[i], 97.5)
    if i < 3:
        d_low = np.percentile(boot_diff[i], 2.5)
        d_high = np.percentile(boot_diff[i], 97.5)
    else:
        d_low = 0
        d_high = 0
    boot_rows.append([boot_names[i], low, high, d_low, d_high])
boot_table = pd.DataFrame(boot_rows, columns=["Model", "R2_low", "R2_high",
                                              "RF_minus_low", "RF_minus_high"])
boot_table = boot_table.round(4)
boot_table.to_csv(os.path.join(RESULTS_DIR, "bootstrap_r2.csv"), index=False)
print(boot_table.to_string(index=False))

# 2) тільки ознаки самого клієнта, без даних кампанії
#    (так модель можна застосувати до нового клієнта до дзвінка)
client_cols = []
for col in X_train.columns:
    first = col.split("_")[0]
    if first in ["Age", "Education", "Job", "Marital Status", "Credit",
                 "Housing Loan", "Personal Loan"]:
        client_cols.append(col)
rf_client = RandomForestRegressor(n_estimators=200, min_samples_leaf=RF_MIN_LEAF,
                                  random_state=42, n_jobs=-1)
rf_client.fit(X_train[client_cols], y_train)
pred_client = rf_client.predict(X_test[client_cols])
m = get_metrics(y_test, pred_client)
print("\nRandom Forest тільки на ознаках клієнта:", len(client_cols), "ознак")
print("MAE =", round(m[0], 1), " RMSE =", round(m[1], 1), " R2 =", round(m[2], 4))
