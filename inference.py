from main import properties_calculation

import numpy as np
import argparse
import yaml
from joblib import load

parser = argparse.ArgumentParser()
parser.add_argument('-password', required=True, help='Пароль, сложность которого необходимо предсказать')
args = parser.parse_args()

strength_in_words = ['слабый', 'средний', 'сильный']

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

scaler = load(config['path_to_scaler'])
clf = load(config['path_to_model'])

def calculate_password_strenght(password: str) -> int:
    sample = properties_calculation(password)
    result = clf.predict(scaler.transform(np.array([sample])))
    return int(result)

strenght = calculate_password_strenght(args.password)

print(f'Сложность пароля {args.password} равна {strenght} - {strength_in_words[strenght]} пароль.')