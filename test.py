# This is a Python script for test classification problem on new data.
from __future__ import annotations
from joblib import load

from main import *

def testing():
    log_file = open('logs/test_logs.txt', 'w+')

    with open('configs/test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['path_to_data'])
    samples, labels = feature_engineering(df)
    scaler = load(config['path_to_scaler'])
    clf = load(config['path_to_model'])
    X_scaled = scaler.transform(samples)

    y_pred = clf.predict(X_scaled)

    calculate_metrics(labels, y_pred, log_file)
    log_file.close()

if __name__ == '__test__':
    testing()
