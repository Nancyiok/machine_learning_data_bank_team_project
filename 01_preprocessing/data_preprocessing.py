import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'raw_data', 'Bank Marketing.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'processed_data')


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Завантажує вихідний CSV-файл із перевіркою існування."""
    if not os.path.exists(file_path):
        print(f"Помилка: файл не знайдено: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path)
    print(f"[1/6] Дані завантажено: {df.shape[0]:,} рядків, {df.shape[1]} стовпців.")
    return df


def clean_and_verify_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищення даних:
    - Перевірка та усунення дублікатів (DataFrame.drop_duplicates());
    - Контроль пропущених значень (DataFrame.isna().sum()).
    """
    initial_rows = len(df)
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"[2/6] Видалено {n_dupes} дублікатів. Залишилось рядків: {len(df):,}.")
    else:
        print(f"[2/6] Перевірка дублікатів: точних копій рядків не виявлено (0).")

    n_missing = df.isna().sum().sum()
    print(f"      Перевірка явних NaN-пропусків: {n_missing}.")
    return df

# ПІДГОТОВКА ДАНИХ ДЛЯ ЗАДАЧІ 1: РЕГРЕСІЯ (REGRESSION)

def prepare_regression_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Підготовка даних для задачі регресії:
    - Цільова змінна: Balance (euros) (числова неперервна характеристика клієнта);
    - Вхідні ознаки: усі ознаки крім Balance та Subscription;
    - Випадковий поділ 80/20 (фіксований random_state=42);
    - Запобігання Data Leakage: fit трансформатора суворо на Train.
    """
    print("\n--- [3/6] Підготовка даних для Задачі 1 (РЕГРЕСІЯ: y = Balance) ---")
    y_reg = df['Balance (euros)'].copy()
    X_reg = df.drop(columns=['Balance (euros)', 'Subscription']).copy()

    # Створення ознаки Previously_Contacted
    X_reg['Previously_Contacted'] = (X_reg['Pdays'] != -1).astype(int)

    # Поділ на Train і Test (для неперервної змінної - звичайний train_test_split)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    print(f"  Поділ вибірки для регресії: Train={X_train_reg.shape[0]:,}, Test={X_test_reg.shape[0]:,}.")

    num_features_reg = [
        'Age', 'Last Contact Day', 'Last Contact Duration',
        'Campaign', 'Pdays', 'Previous', 'Previously_Contacted'
    ]
    ordinal_features = ['Education']
    nominal_features = [
        'Job', 'Marital Status', 'Credit', 'Housing Loan',
        'Personal Loan', 'Contact', 'Last Contact Month', 'Poutcome'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    education_order = [['unknown', 'primary', 'secondary', 'tertiary']]
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_enc', OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features_reg),
            ('ord', ordinal_pipeline, ordinal_features),
            ('nom', nominal_pipeline, nominal_features)
        ],
        remainder='drop'
    )

    # Fit виключно на Train-вибірці регресії
    preprocessor_reg.fit(X_train_reg)
    X_train_reg_trans = preprocessor_reg.transform(X_train_reg)
    X_test_reg_trans = preprocessor_reg.transform(X_test_reg)

    ohe_step = preprocessor_reg.named_transformers_['nom'].named_steps['ohe']
    nom_encoded_names = list(ohe_step.get_feature_names_out(nominal_features))
    all_feature_names_reg = num_features_reg + ordinal_features + nom_encoded_names

    X_train_reg_df = pd.DataFrame(X_train_reg_trans, columns=all_feature_names_reg)
    X_test_reg_df = pd.DataFrame(X_test_reg_trans, columns=all_feature_names_reg)
    print(f"  Трансформація завершена: {X_train_reg_df.shape[1]} ознак після кодування та масштабування.")

    return X_train_reg_df, X_test_reg_df, y_train_reg, y_test_reg

# ПІДГОТОВКА ДАНИХ ДЛЯ ЗАДАЧІ 2: КЛАСИФІКАЦІЯ (CLASSIFICATION)

def prepare_classification_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Підготовка даних для задачі класифікації:
    - Цільова змінна: Subscription (1 -> 0 відмова, 2 -> 1 успіх);
    - Вхідні ознаки: усі інші колонки (включно з Balance);
    - Стратифікований поділ 80/20 (stratify=y);
    - Запобігання Data Leakage: fit трансформатора суворо на Train.
    """
    print("\n--- [4/6] Підготовка даних для Задачі 2 (КЛАСИФІКАЦІЯ: y = Subscription) ---")
    # 1 -> 0 (відмова), 2 -> 1 (підписка на депозит)
    y_class = LabelEncoder().fit_transform(df['Subscription'])

    X_class = df.drop(columns=['Subscription']).copy()
    X_class['Previously_Contacted'] = (X_class['Pdays'] != -1).astype(int)

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    print(f"  Стратифікований поділ для класифікації: Train={X_train_class.shape[0]:,}, Test={X_test_class.shape[0]:,}.")

    num_features_class = [
        'Age', 'Balance (euros)', 'Last Contact Day', 'Last Contact Duration',
        'Campaign', 'Pdays', 'Previous', 'Previously_Contacted'
    ]
    ordinal_features = ['Education']
    nominal_features = [
        'Job', 'Marital Status', 'Credit', 'Housing Loan',
        'Personal Loan', 'Contact', 'Last Contact Month', 'Poutcome'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    education_order = [['unknown', 'primary', 'secondary', 'tertiary']]
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_enc', OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor_class = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features_class),
            ('ord', ordinal_pipeline, ordinal_features),
            ('nom', nominal_pipeline, nominal_features)
        ],
        remainder='drop'
    )

    # Fit виключно на Train-вибірці класифікації
    preprocessor_class.fit(X_train_class)
    X_train_class_trans = preprocessor_class.transform(X_train_class)
    X_test_class_trans = preprocessor_class.transform(X_test_class)

    ohe_step = preprocessor_class.named_transformers_['nom'].named_steps['ohe']
    nom_encoded_names = list(ohe_step.get_feature_names_out(nominal_features))
    all_feature_names_class = num_features_class + ordinal_features + nom_encoded_names

    X_train_class_df = pd.DataFrame(X_train_class_trans, columns=all_feature_names_class)
    X_test_class_df = pd.DataFrame(X_test_class_trans, columns=all_feature_names_class)
    print(f"  Трансформація завершена: {X_train_class_df.shape[1]} ознак після кодування та масштабування.")

    return X_train_class_df, X_test_class_df, y_train_class, y_test_class


def save_all_datasets(
    output_dir: str,
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.Series,
    y_test_reg: pd.Series,
    X_train_class: pd.DataFrame,
    X_test_class: pd.DataFrame,
    y_train_class: np.ndarray,
    y_test_class: np.ndarray
) -> None:
    """Зберігає очищені та трансформовані набори даних для регресії та класифікації."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[5/6] Збереження артефактів у {output_dir}...")

    # 1. Артефакти для регресії
    X_train_reg.to_csv(os.path.join(output_dir, 'X_train_reg.csv'), index=False)
    X_test_reg.to_csv(os.path.join(output_dir, 'X_test_reg.csv'), index=False)
    pd.Series(y_train_reg, name='Balance').to_csv(os.path.join(output_dir, 'y_train_reg.csv'), index=False)
    pd.Series(y_test_reg, name='Balance').to_csv(os.path.join(output_dir, 'y_test_reg.csv'), index=False)
    print("  [+] Файли для регресії збережено: X_train_reg.csv, X_test_reg.csv, y_train_reg.csv, y_test_reg.csv")

    # 2. Артефакти для класифікації
    X_train_class.to_csv(os.path.join(output_dir, 'X_train_class.csv'), index=False)
    X_test_class.to_csv(os.path.join(output_dir, 'X_test_class.csv'), index=False)
    pd.Series(y_train_class, name='Subscription').to_csv(os.path.join(output_dir, 'y_train_class.csv'), index=False)
    pd.Series(y_test_class, name='Subscription').to_csv(os.path.join(output_dir, 'y_test_class.csv'), index=False)
    print("  [+] Файли для класифікації збережено: X_train_class.csv, X_test_class.csv, y_train_class.csv, y_test_class.csv")

    # 3. Файли для зворотної сумісності з попереднім етапом
    X_train_class.to_csv(os.path.join(output_dir, 'X_train_preprocessed.csv'), index=False)
    X_test_class.to_csv(os.path.join(output_dir, 'X_test_preprocessed.csv'), index=False)
    pd.Series(y_train_class, name='Subscription').to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.Series(y_test_class, name='Subscription').to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("  [+] Файли зворотної сумісності (X_train_preprocessed.csv, y_train.csv тощо) оновлено.")


def main():
    df_raw = load_raw_data(INPUT_FILE)
    df_clean = clean_and_verify_data(df_raw)

    # Підготовка для обох задач згідно з п. 3.2 ТЗ
    X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = prepare_regression_data(df_clean)
    X_tr_cls, X_te_cls, y_tr_cls, y_te_cls = prepare_classification_data(df_clean)

    save_all_datasets(
        OUTPUT_DIR,
        X_tr_reg, X_te_reg, y_tr_reg, y_te_reg,
        X_tr_cls, X_te_cls, y_tr_cls, y_te_cls
    )
    print("\n[6/6] Попередня обробка для задач регресії та класифікації успішно завершена!")


if __name__ == '__main__':
    main()
