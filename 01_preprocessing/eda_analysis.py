import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'raw_data', 'Bank Marketing.csv')
IMAGES_DIR = os.path.join(CURRENT_DIR, 'images')
REPORT_FILE = os.path.join(CURRENT_DIR, 'eda_summary_report.txt')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_data(file_path: str) -> pd.DataFrame:
    """Завантажує вхідний датасет та перевіряє його існування."""
    if not os.path.exists(file_path):
        print(f"Помилка: файл не знайдено: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path)
    print(f"[EDA] Дані завантажено: {df.shape[0]} рядків, {df.shape[1]} стовпців.")
    return df


def check_data_cleaning_and_integrity(df: pd.DataFrame, report_path: str) -> dict:
    """
    Виконує повне очищення та перевірку цілісності даних:
    - Пошук дублікатів (DataFrame.duplicated());
    - Підрахунок явних пропусків (DataFrame.isna().sum());
    - Аналіз неявних пропусків (значень 'unknown' у категоріальних ознаках);
    - Повна описова статистика для числових та категоріальних ознак;
    - Оцінка взаємної інформації (Mutual Information) для задач класифікації та регресії.
    """
    print("\n=== [1/5] Перевірка цілісності даних, дублікатів та пропусків ===")
    
    # 1. Пошук дублікатів
    n_duplicates = df.duplicated().sum()
    print(f"  [+] Кількість точних дублікатів (df.duplicated()): {n_duplicates}")

    # 2. Підрахунок явних пропусків (NaN / None)
    missing_series = df.isna().sum()
    total_missing = missing_series.sum()
    print(f"  [+] Кількість явних NaN-пропусків у всіх стовпцях: {total_missing}")

    # 3. Аналіз неявних пропусків (значення 'unknown')
    unknown_stats = {}
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        cnt = (df[col].astype(str).str.lower() == 'unknown').sum()
        if cnt > 0:
            unknown_stats[col] = {'count': cnt, 'pct': (cnt / len(df)) * 100}
            print(f"  [!] Неявні пропуски 'unknown' у '{col}': {cnt:,} ({cnt/len(df)*100:.2f}%)")

    # 4. Описова статистика числових ознак
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc_num = df[num_cols].describe().T
    desc_num['skew'] = df[num_cols].skew()
    desc_num['kurt'] = df[num_cols].kurtosis()

    # 5. Описова статистика категоріальних ознак
    desc_cat = df[cat_cols].describe().T

    # 6. Оцінка аномалій за правилом 1.5 IQR для числових колонок (Слайд 5)
    iqr_stats = {}
    for col in num_cols:
        if col == 'Subscription':
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        iqr_stats[col] = {
            'q1': q1, 'q3': q3, 'iqr': iqr,
            'lower_bound': lower, 'upper_bound': upper,
            'outliers_cnt': outliers_cnt,
            'outliers_pct': (outliers_cnt / len(df)) * 100
        }

    # 7. Взаємна інформація
    # Таргет класифікації: Subscription (бінарна 0/1)
    y_class = (df['Subscription'] == 2).astype(int)
    numeric_features = [c for c in num_cols if c != 'Subscription']
    mi_class = mutual_info_classif(df[numeric_features], y_class, random_state=42)
    mi_class_series = pd.Series(mi_class, index=numeric_features).sort_values(ascending=False)

    # Таргет регресії: Balance (euros) (числова)
    y_reg = df['Balance (euros)']
    features_reg = [c for c in numeric_features if c != 'Balance (euros)']
    mi_reg = mutual_info_regression(df[features_reg], y_reg, random_state=42)
    mi_reg_series = pd.Series(mi_reg, index=features_reg).sort_values(ascending=False)

    # Запис звіту у файл
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Розмірність датасету: {df.shape[0]:,} рядків, {df.shape[1]} стовпців\n")
        f.write(f"Точні дублікати (df.duplicated().sum()): {n_duplicates}\n")
        f.write(f"Явні пропуски (df.isna().sum()): {total_missing}\n\n")

        f.write("1. АНАЛІЗ НЕЯВНИХ ПРОПУСКІВ ('unknown'):\n")
        for col, info in unknown_stats.items():
            f.write(f"  - {col}: {info['count']:,} ({info['pct']:.2f}%)\n")
        f.write("  Примітка: у 'Poutcome' 81.7% значень 'unknown' через відсутність попереднього контакту (Pdays = -1).\n\n")

        f.write("2. ОПИСОВА СТАТИСТИКА ЧИСЛОВИХ ОЗНАК:\n")
        f.write(desc_num.to_string())
        f.write("\n\n")

        f.write("3. ОПИСОВА СТАТИСТИКА КАТЕГОРІАЛЬНИХ ОЗНАК:\n")
        f.write(desc_cat.to_string())
        f.write("\n\n")

        f.write("4. АНАЛІЗ АНОМАЛІЙ ЗА ПРАВИЛОМ 1.5 IQR :\n")
        for col, s in iqr_stats.items():
            f.write(f"  - {col}: Q1={s['q1']:.1f}, Q3={s['q3']:.1f}, IQR={s['iqr']:.1f} | Межі: [{s['lower_bound']:.1f}; {s['upper_bound']:.1f}] | Викидів: {s['outliers_cnt']:,} ({s['outliers_pct']:.2f}%)\n")
        f.write("  Висновок: викиди у 'Balance' та 'Duration' є валідними спостереженнями (заможні клієнти, тривалі успішні переговори); рекомендовано RobustScaler.\n\n")

        f.write("5. ВЗАЄМНА ІНФОРМАЦІЯ (Mutual Information) З ТАРГЕТАМИ:\n")
        f.write("  А) Для задачі Класифікації (y = Subscription):\n")
        for col, val in mi_class_series.items():
            f.write(f"     * {col}: {val:.4f}\n")
        f.write("  Б) Для задачі Регресії (y = Balance (euros)):\n")
        for col, val in mi_reg_series.items():
            f.write(f"     * {col}: {val:.4f}\n")

    print(f"  [+] Повний аналітичний звіт збережено у: {report_path}")
    return {
        'duplicates': n_duplicates,
        'missing': total_missing,
        'unknowns': unknown_stats,
        'iqr_stats': iqr_stats,
        'mi_class': mi_class_series,
        'mi_reg': mi_reg_series
    }


def plot_class_imbalance(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 1: Дисбаланс цільової змінної Subscription (88/12)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    target_counts = df['Subscription'].value_counts()
    labels = ['Відмова (Клас 1)', 'Згода (Клас 2)']
    values = [target_counts.get(1, 0), target_counts.get(2, 0)]
    colors = ['#2b5c8f', '#e26d5c']

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor='black', linewidth=1)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_title('Дисбаланс класів цільової змінної Subscription (88.3% проти 11.7%)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Кількість спостережень (клієнтів)', fontsize=11)
    ax.set_ylim(0, max(values) * 1.18)

    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_class_imbalance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 1 (Дисбаланс класів): {output_path}")
    return output_path


def plot_numerical_distributions(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 2: Розподіли основних числових колонок (фокус на 97–99% клієнтів: Медіана vs Середнє)."""
    configs = [
        {
            'col': 'Age',
            'title': 'Вік клієнта (Age)',
            'unit': 'р.',
            'xlim': (18, 75),
            'bins': 28,
            'kde': True,
            'discrete': False,
            'val_fmt': '.1f',
            'tail_note': 'Максимум: 95 років\n(99.4% клієнтів віком до 75 р.;\nрозподіл близький до нормального)'
        },
        {
            'col': 'Balance (euros)',
            'title': 'Баланс на рахунку (Balance)',
            'unit': '€',
            'xlim': (-2000, 7500),
            'bins': 45,
            'kde': True,
            'discrete': False,
            'val_fmt': '.0f',
            'tail_note': 'Хвіст до 102 127 € (10.5% аномалій)\n(багаті вкладники зсувають\nсереднє далеко вправо від медіани!)'
        },
        {
            'col': 'Last Contact Duration',
            'title': 'Тривалість останньої розмови (Duration)',
            'unit': 'с',
            'xlim': (0, 1200),
            'bins': 40,
            'kde': True,
            'discrete': False,
            'val_fmt': '.0f',
            'tail_note': 'Хвіст до 4 918 с (~82 хв)\n(98.8% дзвінків до 20 хв;\nдовгі розмови ведуть до успіху)'
        },
        {
            'col': 'Campaign',
            'title': 'Кількість контактів у кампанії (Campaign)',
            'unit': 'дзв.',
            'xlim': (0.3, 10.7),
            'bins': 10,
            'kde': False,
            'discrete': True,
            'val_fmt': '.1f',
            'tail_note': 'Хвіст до 63 дзвінків\n(97.4% клієнтів мають ≤ 10 дзвінків;\nмедіана = 2 контакти)'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()

    for i, cfg in enumerate(configs):
        ax = axes[i]
        col = cfg['col']

        data_in_range = df[(df[col] >= cfg['xlim'][0]) & (df[col] <= cfg['xlim'][1])][col]

        if cfg['discrete']:
            sns.histplot(
                data_in_range, discrete=True, ax=ax,
                color='#1d3557', stat='count', alpha=0.55, edgecolor='#1d3557',
                shrink=0.85
            )
            ax.set_xticks(range(1, 11))
        else:
            sns.histplot(
                data_in_range, kde=cfg['kde'], ax=ax,
                color='#1d3557', bins=cfg['bins'], stat='count', alpha=0.55, edgecolor='#1d3557'
            )

        mean_val = df[col].mean()
        median_val = df[col].median()

        fmt = cfg['val_fmt']
        dur_extra_med = f" (~{median_val/60:.1f} хв)" if col == 'Last Contact Duration' else ''
        dur_extra_mean = f" (~{mean_val/60:.1f} хв)" if col == 'Last Contact Duration' else ''

        ax.axvline(median_val, color='#e63946', linestyle='--', linewidth=2.5,
                   label=f"Медіана: {median_val:{fmt}} {cfg['unit']}{dur_extra_med}")
        ax.axvline(mean_val, color='#2a9d8f', linestyle=':', linewidth=2.5,
                   label=f"Середнє: {mean_val:{fmt}} {cfg['unit']}{dur_extra_mean}")

        ax.text(
            0.97, 0.70, cfg['tail_note'],
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3b0', edgecolor='#e09f3e', alpha=0.9)
        )

        ax.set_xlim(cfg['xlim'])
        ax.set_title(cfg['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"{col} ({cfg['unit']})", fontsize=10, fontweight='bold')
        ax.set_ylabel('Кількість клієнтів', fontsize=10)
        ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

    plt.suptitle('Розподіли основних числових ознак (Фокус на 97–99% вибірки: Медіана vs Середнє)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_numerical_distributions.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 2 (Розподіли числових ознак): {output_path}")
    return output_path


def plot_outliers_iqr(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 3: Діаграми розмаху (Boxplot) та правило 1.5 IQR """
    configs = [
        {
            'col': 'Age',
            'title': 'Вік клієнта (Age)',
            'unit': 'років',
            'xlim': (10, 95),
            'clamp_min': 18,
            'tail_note': 'Максимум: 95 років\n(пенсіонери, не помилка)'
        },
        {
            'col': 'Balance (euros)',
            'title': 'Баланс на рахунку (Balance)',
            'unit': '€',
            'xlim': (-2500, 7500),
            'clamp_min': None,
            'tail_note': 'Хвіст до 102 127 €\n(VIP-вкладники, 10.5% вибірки)'
        },
        {
            'col': 'Last Contact Duration',
            'title': 'Тривалість розмови (Duration)',
            'unit': 'с',
            'xlim': (0, 1100),
            'clamp_min': 0,
            'tail_note': 'Хвіст до 4 918 с (~82 хв)\n(тривалі результативні розмови)'
        },
        {
            'col': 'Campaign',
            'title': 'Кількість дзвінків (Campaign)',
            'unit': 'дзв.',
            'xlim': (0, 14),
            'clamp_min': 1,
            'tail_note': 'Максимум: до 63 дзвінків\n(інтенсивна кампанія)'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()

    for i, cfg in enumerate(configs):
        ax = axes[i]
        col = cfg['col']
        s = df[col]

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if cfg['clamp_min'] is not None and lower < cfg['clamp_min']:
            display_lower = None
        else:
            display_lower = lower

        outliers = ((s < lower) | (s > upper)).sum()
        pct = (outliers / len(s)) * 100

        ax.boxplot(
            s, orientation='horizontal', patch_artist=True, widths=0.45,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='#e63946', markeredgecolor='none', markersize=4, alpha=0.35),
            medianprops=dict(color='#1d3557', linewidth=2.5),
            boxprops=dict(facecolor='#a8dadc', edgecolor='#1d3557', linewidth=1.5),
            whiskerprops=dict(color='#1d3557', linewidth=1.5),
            capprops=dict(color='#1d3557', linewidth=1.5)
        )

        ax.axvspan(cfg['xlim'][0], upper, color='#2a9d8f', alpha=0.18, label='Зона норми (до 1.5 IQR)')
        ax.axvspan(upper, cfg['xlim'][1], color='#e63946', alpha=0.18, label=f"Зона аномалій ({pct:.1f}%)")
        ax.axvline(upper, color='#b7094c', linestyle='--', linewidth=2, label=f"Поріг 1.5 IQR: {upper:.0f} {cfg['unit']}")

        if display_lower is not None:
            ax.axvline(display_lower, color='#b7094c', linestyle=':', linewidth=1.5, label=f"Нижня межа: {display_lower:.0f} {cfg['unit']}")

        med = s.median()
        ax.text(med, 1.28, f"Медіана: {med:.0f}", ha='center', fontsize=9, fontweight='bold', color='#1d3557')

        ax.text(
            0.97, 0.22, cfg['tail_note'],
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3b0', edgecolor='#e09f3e', alpha=0.9)
        )

        ax.set_xlim(cfg['xlim'])
        ax.set_yticks([])
        ax.set_xlabel(f"{col} ({cfg['unit']})", fontsize=10, fontweight='bold')
        ax.set_title(f"{cfg['title']} | Викидів за 1.5 IQR: {outliers:,} ({pct:.1f}%)", fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    plt.suptitle('Виявлення аномалій за правилом 1.5 IQR (Фокусний масштаб на розподілі більшості клієнтів)', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_outliers_iqr_boxplots.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 3 (Викиди 1.5 IQR): {output_path}")
    return output_path


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 4: Теплова карта кореляцій Пірсона для числових колонок."""
    num_cols = ['Age', 'Balance (euros)', 'Last Contact Day', 'Last Contact Duration', 'Campaign', 'Pdays', 'Previous']
    df_corr = df[num_cols].copy()
    df_corr['Subscription (0/1)'] = (df['Subscription'] == 2).astype(int)

    corr_matrix = df_corr.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='vlag', center=0,
        square=True, linewidths=0.7, cbar_kws={'shrink': 0.8},
        ax=ax, annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )

    ax.set_title('Матриця кореляцій Пірсона для числових ознак та цільової змінної', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    output_path = os.path.join(output_dir, '04_correlation_matrix.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 4 (Матриця кореляцій): {output_path}")
    return output_path


def plot_categorical_distributions(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 5: Розподіли ключових категоріальних ознак (Education, Job, Marital, Housing Loan)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes = axes.flatten()

    # 1. Education (Порядкова ознака)
    ax1 = axes[0]
    edu_order = ['primary', 'secondary', 'tertiary', 'unknown']
    counts_edu = df['Education'].value_counts().reindex(edu_order).fillna(0)
    bars1 = ax1.bar(edu_order, counts_edu, color='#2a9d8f', edgecolor='black', alpha=0.85)
    ax1.set_title('Рівень освіти (Education) - Порядкова ознака', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Кількість клієнтів')
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax1.set_ylim(0, max(counts_edu) * 1.18)

    # 2. Marital Status
    ax2 = axes[1]
    counts_mar = df['Marital Status'].value_counts()
    bars2 = ax2.bar(counts_mar.index, counts_mar.values, color='#457b9d', edgecolor='black', alpha=0.85)
    ax2.set_title('Сімейний стан (Marital Status)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Кількість клієнтів')
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax2.set_ylim(0, max(counts_mar.values) * 1.18)

    # 3. Housing Loan
    ax3 = axes[2]
    counts_hl = df['Housing Loan'].value_counts()
    bars3 = ax3.bar(counts_hl.index, counts_hl.values, color=['#e63946', '#1d3557'], edgecolor='black', alpha=0.85)
    ax3.set_title('Наявність іпотеки (Housing Loan)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Кількість клієнтів')
    for b in bars3:
        h = b.get_height()
        ax3.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", (b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax3.set_ylim(0, max(counts_hl.values) * 1.18)

    # 4. Job (Топ-6 найпоширеніших професій)
    ax4 = axes[3]
    top_jobs = df['Job'].value_counts().head(6)
    bars4 = ax4.barh(top_jobs.index[::-1], top_jobs.values[::-1], color='#e09f3e', edgecolor='black', alpha=0.85)
    ax4.set_title('Топ-6 професій клієнтів (Job)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Кількість клієнтів')
    for b in bars4:
        w = b.get_width()
        ax4.annotate(f" {w:,} ({w/len(df)*100:.1f}%)", (w, b.get_y() + b.get_height()/2),
                     va='center', fontsize=8.5, fontweight='bold')
    ax4.set_xlim(0, max(top_jobs.values) * 1.25)

    plt.suptitle('Розподіли ключових категоріальних ознак датасету Bank Marketing', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_categorical_distributions.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 5 (Категоріальні ознаки): {output_path}")
    return output_path


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = load_data(INPUT_FILE)

    print("\n=== Генерація аналітики та графіків EDA ===")
    check_data_cleaning_and_integrity(df, REPORT_FILE)
    plot_class_imbalance(df, IMAGES_DIR)
    plot_numerical_distributions(df, IMAGES_DIR)
    plot_outliers_iqr(df, IMAGES_DIR)
    plot_correlation_matrix(df, IMAGES_DIR)
    plot_categorical_distributions(df, IMAGES_DIR)

    print(f"\nУсі графіки та звіти EDA успішно сформовано у:\n  -> Звіт: {REPORT_FILE}\n  -> Графіки: {IMAGES_DIR}")


if __name__ == '__main__':
    main()
