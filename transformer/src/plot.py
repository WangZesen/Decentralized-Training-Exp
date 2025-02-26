import os
import re
import sys
import tomllib
import numpy as np
from typing import Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.conf import Train as TrainConfig
from loguru import logger
logger.remove()
logger.add(sys.stdout)
sns.set_theme(style='whitegrid', font_scale=1.4)

def read_test_log(exp_dir: str):
    def _load_toml(file_path: str):
        with open(file_path, 'rb') as f:
            return tomllib.load(f)
    num_workers = 0
    test_log = pd.read_csv(os.path.join(exp_dir, 'test_log.csv'), sep=',')
    with open(os.path.join(exp_dir, 'train_cfg.dump.toml'), 'r') as f:
        dumped_train_cfg = f.read()
        matches = re.findall('world_size = .*\n', dumped_train_cfg)
        num_workers = int(matches[0].split(' ')[2])
    train_cfg = TrainConfig(**_load_toml(os.path.join(exp_dir, 'train_cfg.dump.toml')))
    label = f'transformer - {num_workers} workers{" - amp" if train_cfg.use_amp else ""}'
    test_log['label'] = label
    return test_log, test_log['BLEU Score'].to_list()[-1], test_log['time'].to_list()[-1]

def plot(logs: pd.DataFrame, x: str, y: str, img_dir: str, xscale: str='linear', yscale: str='linear'):
    plt.clf()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=logs, x=x, y=y, style='label', hue='label', errorbar=('sd', 2))
    if y == 'BLEU Score':
        # plot baseline
        plt.axhline(y=0.273, color='r', linestyle='--', label='baseline')
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlim(left=0)
    # if y == 'BLEU Score':
    #     plt.ylim(bottom=0.2)
    plt.tight_layout()
    plt.savefig(img_dir, dpi=300, bbox_inches='tight')

def main():
    exp_dirs = sys.argv[1:]

    time_values: Dict[str, list] = {}
    last_bleu_values: Dict[str, list] = {}
    total_train_time_values: Dict[str, list] = {}
    for exp_dir in exp_dirs:
        test_log, last_bleu_score, total_train_time = read_test_log(exp_dir)
        for label in test_log['label'].unique():
            if label not in time_values:
                time_values[label] = []
                last_bleu_values[label] = []
                total_train_time_values[label] = []
            time_values[label].extend(test_log[test_log['label'] == label]['time'].unique().tolist())
            last_bleu_values[label].append(last_bleu_score)
            total_train_time_values[label].append(total_train_time)
    for label in time_values:
        time_values[label] = sorted(list(set(time_values[label])))
    
    interp_logs = []
    for exp_dir in exp_dirs:
        test_log, _, _ = read_test_log(exp_dir)
        interpolated = pd.DataFrame(columns=test_log.columns)

        label = test_log['label'].unique()[0]
        x = test_log['time'].to_list()
        y_bleu = test_log['BLEU Score'].to_list()
        y_val_loss = test_log['val_loss'].to_list()

        f_bleu = interp1d(x, y_bleu, kind='linear', fill_value='extrapolate')
        f_val_loss = interp1d(x, y_val_loss, kind='linear', fill_value='extrapolate')

        interp_bleu = f_bleu(time_values[label])
        interp_val_loss = f_val_loss(time_values[label])

        interpolated['time'] = time_values[label]
        interpolated['BLEU Score'] = interp_bleu
        interpolated['val_loss'] = interp_val_loss
        interpolated['label'] = label
        interp_logs.append(interpolated)

    interp_log = pd.concat(interp_logs)
    test_log = pd.concat([read_test_log(exp_dir)[0] for exp_dir in exp_dirs])

    os.makedirs('image', exist_ok=True)
    plot(interp_log, 'time', 'BLEU Score', 'image/time_vs_bleu.png')
    plot(test_log, 'step', 'BLEU Score', 'image/step_vs_bleu.png')
    plot(interp_log, 'time', 'val_loss', 'image/time_vs_valloss.png')
    plot(test_log, 'step', 'val_loss', 'image/step_vs_valloss.png')

    logger.info(f'Plots generated at image/ directory')

    for label in last_bleu_values:
        # log the last bleu score with 2x standard deviation
        _bleu_score = np.array(last_bleu_values[label])
        _train_time = np.array(total_train_time_values[label]) / 60 / 60
        logger.info(f'{label} - last BLEU score: {np.mean(_bleu_score):.4f} ± {2 * np.std(_bleu_score):.4f}')
        logger.info(f'{label} - total training time: {np.mean(_train_time):.4f} ± {2 * np.std(_train_time):.4f} hours')


if __name__ == '__main__':
    main()
