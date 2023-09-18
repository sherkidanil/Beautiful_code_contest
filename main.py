# This is a Python script for classification problem solve using XGBoost.
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from joblib import dump

def properties_calculation(password: str) -> List[int]:
    """Возвращает параметры строки.
    Args:
        password (str): пароль, чьи параметры необходимо посчитать.
    Returns:
        List[int]: список состоящий из [длина, число заглавных букв, число строчных букв,
                                        число букв, число симвлов].
    """
    length = len(password)
    uppercase_num = sum([1 for char in password if char.isupper()])
    lowercase_num = sum([1 for char in password if char.islower()])
    number_num = sum([1 for char in password if char.isdigit()])
    symbol_num = sum([1 for char in password if char.isalpha()])
    return [length, uppercase_num, lowercase_num, number_num, symbol_num]

def logging(log_file: str, output: str|Any):
    """Простой логгер для записи выборочного стандартного вывода.
    Args:
        log_file (str): путь до лог-файла, в который мы записываем наши логи.
        output (str | Any): что-то, что нужно записать в логи.
                            Конвертируемо в строку.
    """
    print(output)
    log_file.write(str(output) + '\n')

def feature_engineering(df: pd.DataFrame) -> Tuple[np.array]:
    """Преобразуем исходный датафрейм в два массива с наблюдениями и метками.
    Args:
        df (pd.DataFrame): исходный набор данных для преобразования.
    Returns:
        Tuple[np.array]: кортеж для распаковки с преобразоваными наблюдениями и их метками.
    """
    samples = np.array([properties_calculation(x) for x in df.password])
    labels = df['strength'].to_numpy()
    return (samples, labels)

def calculate_metrics(y_true: np.array, y_pred: np.array, log_file: str):
    """Функция для расчета основных метрик.
    Args:
        y_true (np.array): верные результаты, с котороыми сравниваем предсказания;
        y_pred (np.array): предсказания модели;
        log_file (str): путь к лог-файлу, куда сохраним результаты расчета метрик.
    """
    logging(log_file, f"Accuracy: {accuracy_score(y_true, y_pred)}")
    logging(log_file, f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
    logging(log_file, f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
    logging(log_file, f"F1-score: {f1_score(y_true, y_pred, average='weighted')}")
    logging(log_file, "Confusion Matrix:")
    logging(log_file, confusion_matrix(y_true, y_pred))

def main():
    log_file = open('logs/logs.txt', 'w+')

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['path_to_data'])
    samples, labels = feature_engineering(df)
    X_train, X_test, y_train, y_test =  train_test_split(samples, labels,
                                                         test_size=config['test_size'],
                                                         random_state=config['random_state'])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    dump(scaler, config['path_to_scaler'])
    X_test_scaled = scaler.transform(X_test)
    clf = XGBClassifier(objective=f'multi:{config["objective"]}',
                        num_class=3,
                        max_depth=config['max_depth'],
                        learning_rate=config['learning_rate'],
                        n_estimators=config['n_estimators'],
                        nthread=-1,
                        seed=config['random_state'])
    clf.fit(X_train_scaled, y_train)
    dump(clf, config['path_to_model'])
    y_pred = clf.predict(X_test_scaled)
    calculate_metrics(y_test, y_pred, log_file)

    log_file.close()

if __name__ == '__main__':
    main()
