from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Завантаження даних
X_train = pd.read_csv("data_claster/X_train_preprocessed.csv")
X_test = pd.read_csv("data_claster/X_test_preprocessed.csv")

# 2. Перевірка розмірів
print("Розмір навчальних даних:", X_train.shape)
print("Розмір тестових даних:", X_test.shape)

# 3. Перегляд перших 5 рядків
print("\nПерші 5 рядків:")
print(X_train.head())

# 4. Перегляд назв ознак
print("\nНазви ознак:")
print(X_train.columns.tolist())

# 5. Перевірка пропущених значень
print("\nКількість пропущених значень:")
print(X_train.isna().sum())

# 6. Перевірка типів даних
print("\nТипи даних:")
print(X_train.dtypes)

# 7. Основні статистичні характеристики
print("\nОписова статистика:")
print(X_train.describe())



# 8. Об'єднання даних
X_all = pd.concat([X_train, X_test], ignore_index=True)

print("\nРозмір об'єднаних даних:", X_all.shape)

# 9. Вибір ознак для кластеризації
features = [
    "Age",
    "Balance (euros)",
    "Education",
    "Housing Loan_yes",
    "Personal Loan_yes",
    "Marital Status_married",
    "Marital Status_single"
]

X_cluster = X_all[features].copy()

print("\nОбрані ознаки:")
print(X_cluster.columns.tolist())

print("\nСтатистика обраних ознак:")
print(X_cluster.describe())







# 10. Два варіанти даних
X_original = X_cluster.copy()
X_limited = X_cluster.copy()

balance_limit = X_original["Balance (euros)"].quantile(0.95)

X_limited["Balance (euros)"] = (
    X_limited["Balance (euros)"].clip(upper=balance_limit)
)

print("95-й процентиль балансу:", balance_limit)

datasets = {
    "Без обмеження": X_original,
    "З обмеженням 95%": X_limited
}

# 11. Дослідження кількості кластерів
k_values = range(2, 9)
results = {}

for name, X in datasets.items():
    print(f"\nЕксперимент: {name}")
    rows = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=3000,
            random_state=42
        )

        rows.append({
            "k": k,
            "Inertia": model.inertia_,
            "Silhouette": score
        })

        print(f"k={k}, інерція={model.inertia_:.2f}, "
              f"силует={score:.4f}, "
              f"розміри={np.bincount(labels)}")

    results[name] = pd.DataFrame(rows)

# 12. Порівняння інерції та силуету
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row, (name, table) in enumerate(results.items()):
    axes[row, 0].plot(
        table["k"], table["Inertia"], marker="o"
    )
    axes[row, 0].set_title(f"Метод ліктя: {name}")
    axes[row, 0].set_xlabel("Кількість кластерів")
    axes[row, 0].set_ylabel("Інерція")
    axes[row, 0].grid(True)

    axes[row, 1].plot(
        table["k"], table["Silhouette"], marker="o"
    )
    axes[row, 1].set_title(f"Коефіцієнт силуету: {name}")
    axes[row, 1].set_xlabel("Кількість кластерів")
    axes[row, 1].set_ylabel("Silhouette Score")
    axes[row, 1].grid(True)

plt.tight_layout()
plt.savefig("comparison_k.png", dpi=200)
plt.show()

# 13. Візуалізація кластерів через PCA
# Поки що для порівняння використовуємо k=3

# 13. Візуалізація кластерів через PCA
k_vis = 3

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rng = np.random.default_rng(42)
sample_idx = rng.choice(
    len(X_cluster),
    size=min(6000, len(X_cluster)),
    replace=False
)

for ax, (name, X) in zip(axes, datasets.items()):
    model = KMeans(
        n_clusters=k_vis,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # Зменшення розмірності до двох компонент
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Перетворення координат центрів кластерів
    centers_pca = pca.transform(
        pd.DataFrame(
            model.cluster_centers_,
            columns=X.columns
        )
    )

    # Відображення точок кластерів
    for cluster in range(k_vis):
        idx = sample_idx[labels[sample_idx] == cluster]

        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=8,
            alpha=0.5,
            label=f"Кластер {cluster}"
        )

    # Відображення центрів кластерів
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        marker="X",
        s=150,
        label="Центри кластерів"
    )

    ax.set_title(f"{name}, k={k_vis}")
    ax.set_xlabel("Головна компонента 1")
    ax.set_ylabel("Головна компонента 2")
    ax.legend()
    ax.grid(True)

    print(f"\nЕксперимент: {name}")
    print("Розміри кластерів:", np.bincount(labels))
    print(
        "Частка поясненої дисперсії:",
        pca.explained_variance_ratio_.sum()
    )

plt.tight_layout()
plt.savefig("clusters_pca.png", dpi=200)
plt.show()



# Аналіз 4 кластерів після обмеження балансу
model4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels4 = model4.fit_predict(X_limited)

profile4 = X_original.copy()
profile4["Cluster"] = labels4

print("\nХарактеристики 4 кластерів:")
print(
    profile4.groupby("Cluster").agg(
        Count=("Age", "size"),
        Age_median=("Age", "median"),
        Balance_median=("Balance (euros)", "median"),
        Education_median=("Education", "median"),
        Housing_loan=("Housing Loan_yes", "mean"),
        Personal_loan=("Personal Loan_yes", "mean")
    ).round(3).to_string()
)


# Візуалізація 4 кластерів після обмеження
pca4 = PCA(n_components=2)
X_pca4 = pca4.fit_transform(X_limited)

centers4 = pca4.transform(
    pd.DataFrame(
        model4.cluster_centers_,
        columns=X_limited.columns
    )
)

plt.figure(figsize=(10, 7))

for cluster in range(4):
    idx = sample_idx[labels4[sample_idx] == cluster]

    plt.scatter(
        X_pca4[idx, 0],
        X_pca4[idx, 1],
        s=8,
        alpha=0.5,
        label=f"Кластер {cluster}"
    )

plt.scatter(
    centers4[:, 0],
    centers4[:, 1],
    c="black",
    marker="X",
    s=180,
    label="Центри кластерів"
)

plt.title("K-Means: 4 кластери, обмеження 95%")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("clusters_k4_limited.png", dpi=200)
plt.show()


# Відновлення початкових значень віку та балансу
profile_real = X_original.copy()

profile_real["Age_years"] = (
    profile_real["Age"] * 15 + 39
)

profile_real["Balance_euros"] = (
    profile_real["Balance (euros)"] * 1356.25 + 451
)

# Мітки кластерів із моделі k=4
profile_real["Cluster"] = labels4

# Характеристики кластерів у початкових одиницях
summary = profile_real.groupby("Cluster").agg(
    Count=("Age_years", "size"),
    Age_median=("Age_years", "median"),
    Balance_median=("Balance_euros", "median"),
    Housing_loan=("Housing Loan_yes", "mean"),
    Personal_loan=("Personal Loan_yes", "mean")
)

summary["Housing_loan"] *= 100
summary["Personal_loan"] *= 100

print("\nХарактеристики кластерів:")
print(summary.round(2).to_string())


# 17. Характеристики кластерів без обмеження, k=3

model3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels3 = model3.fit_predict(X_original)

profile3 = X_original.copy()

# Повернення до початкових одиниць
profile3["Age_years"] = profile3["Age"] * 15 + 39
profile3["Balance_euros"] = (
    profile3["Balance (euros)"] * 1356.25 + 451
)

profile3["Cluster"] = labels3

summary3 = profile3.groupby("Cluster").agg(
    Count=("Age_years", "size"),
    Age_median=("Age_years", "median"),
    Balance_median=("Balance_euros", "median"),
    Housing_loan=("Housing Loan_yes", "mean"),
    Personal_loan=("Personal Loan_yes", "mean")
)

summary3["Housing_loan"] *= 100
summary3["Personal_loan"] *= 100

summary3["Share_%"] = (
    summary3["Count"] / len(profile3) * 100
)

print("\nБЕЗ ОБМЕЖЕННЯ, k=3")
print(summary3.round(2).to_string())

# 18. Підсумкова таблиця для варіанта з обмеженням

summary["Share_%"] = (
    summary["Count"] / len(profile_real) * 100
)

print("\nЗ ОБМЕЖЕННЯМ 95%, k=4")
print(summary.round(2).to_string())

# 19. Збереження таблиць
summary3.to_csv(
    "clusters_original_k3.csv",
    encoding="utf-8-sig"
)

summary.to_csv(
    "clusters_limited_k4.csv",
    encoding="utf-8-sig"
)



# 20. Завантаження інформації про депозити
y_train = pd.read_csv("data_claster/y_train.csv")
y_test = pd.read_csv("data_claster/y_test.csv")

y_all = pd.concat(
    [y_train, y_test],
    ignore_index=True
)["Subscription"].astype(int)

# Перевірка відповідності кількості записів
assert len(y_all) == len(X_limited)
print("\nРозподіл депозитів:")
print(y_all.value_counts().sort_index())

# 21. Кластеризація окремо для двох класів
k_deposit = 4

# Єдина система координат PCA для порівняння
pca_deposit = PCA(n_components=2)
pca_deposit.fit(X_limited)

fig, axes = plt.subplots(
    1, 2, figsize=(14, 6),
    sharex=True, sharey=True
)

for ax, deposit_value in zip(axes, [0, 1]):

    # Вибір клієнтів за фактом відкриття депозиту
    X_group = X_limited.loc[
        y_all == deposit_value
    ]

    # Навчання K-Means лише на обраній групі
    model = KMeans(
        n_clusters=k_deposit,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_group)

    score = silhouette_score(
        X_group,
        labels,
        sample_size=3000,
        random_state=42
    )

    print(f"\nДепозит = {deposit_value}")
    print("Кількість клієнтів:", len(X_group))
    print("Коефіцієнт силуету:", round(score, 4))
    print("Розміри кластерів:", np.bincount(labels))

    # Профілі кластерів у початкових одиницях
    profile = X_original.loc[X_group.index].copy()
    profile["Cluster"] = labels
    profile["Age_years"] = profile["Age"] * 15 + 39
    profile["Balance_euros"] = (
        profile["Balance (euros)"] * 1356.25 + 451
    )

    summary_deposit = profile.groupby("Cluster").agg(
        Count=("Age_years", "size"),
        Age_median=("Age_years", "median"),
        Balance_median=("Balance_euros", "median"),
        Housing_loan=("Housing Loan_yes", "mean"),
        Personal_loan=("Personal Loan_yes", "mean")
    )

    print(summary_deposit.round(2).to_string())

    # Відбір точок для візуалізації
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        X_group.index.to_numpy(),
        size=min(4000, len(X_group)),
        replace=False
    )

    X_sample = X_group.loc[sample_idx]
    label_series = pd.Series(
        labels, index=X_group.index
    )
    sample_labels = label_series.loc[sample_idx].to_numpy()

    X_pca = pca_deposit.transform(X_sample)

    for cluster in range(k_deposit):
        mask = sample_labels == cluster

        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8,
            alpha=0.5,
            label=f"Кластер {cluster}"
        )

    # Центри кластерів
    centers = pca_deposit.transform(
        pd.DataFrame(
            model.cluster_centers_,
            columns=X_group.columns
        )
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        marker="X",
        s=160,
        label="Центри"
    )

    title = (
        "Не відкрили депозит"
        if deposit_value == 0
        else "Відкрили депозит"
    )

    ax.set_title(f"{title}, k={k_deposit}")
    ax.set_xlabel("Головна компонента 1")
    ax.set_ylabel("Головна компонента 2")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("clusters_by_deposit.png", dpi=200)
plt.show()


# 22. Аналіз відкриття депозитів у 4 основних кластерах

cluster_deposits = pd.DataFrame({
    "Cluster": labels4,
    "Subscription": y_all.to_numpy()
})

# Кількість клієнтів та відкритих депозитів
deposit_stats = cluster_deposits.groupby("Cluster").agg(
    Clients=("Subscription", "size"),
    Opened=("Subscription", "sum"),
    Rate=("Subscription", "mean")
)

deposit_stats["Not_opened"] = (
    deposit_stats["Clients"] - deposit_stats["Opened"]
)

# Частка відкритих депозитів у відсотках
deposit_stats["Rate_%"] = deposit_stats["Rate"] * 100

# Загальна частка відкритих депозитів
overall_rate = y_all.mean() * 100

print("\nВідкриття депозитів за кластерами:")
print(deposit_stats.round(2).to_string())

print(f"\nЗагальна частка відкритих депозитів: {overall_rate:.2f}%")

# 23. Візуалізація частки відкритих депозитів
plt.figure(figsize=(9, 6))

bars = plt.bar(
    deposit_stats.index.astype(str),
    deposit_stats["Rate_%"]
)

plt.axhline(
    y=overall_rate,
    color="red",
    linestyle="--",
    label=f"Середня частка: {overall_rate:.2f}%"
)

for bar, value in zip(bars, deposit_stats["Rate_%"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}%",
        ha="center"
    )

plt.title("Частка відкритих депозитів за кластерами")
plt.xlabel("Кластер")
plt.ylabel("Частка відкритих депозитів (%)")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("deposit_rate_by_cluster.png", dpi=200)
plt.show()

# 24. Збереження результатів
deposit_stats.to_csv(
    "deposit_stats_by_cluster.csv",
    encoding="utf-8-sig"
)
