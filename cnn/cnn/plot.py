import os
import re
import sys
import tomllib
import numpy as np
from typing import Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.interpolate import interp1d
from cnn.conf import Train as TrainConfig
from loguru import logger
logger.remove()
logger.add(sys.stdout)
sns.set_theme(style='whitegrid', font_scale=1.4)


def read_train_log(exp_dir: str) -> Tuple[pd.DataFrame, float, float, float]:
    def _load_toml(file_path: str):
        with open(file_path, 'rb') as f:
            return tomllib.load(f)
    num_workers = 0
    test_log = pd.read_csv(os.path.join(exp_dir, 'train_log.csv'), sep=',')
    with open(os.path.join(exp_dir, 'train_cfg.dump.toml'), 'r') as f:
        dumped_train_cfg = f.read()
        matches = re.findall('world_size = .*\n', dumped_train_cfg)
        num_workers = int(matches[0].split(' ')[2])
    train_cfg = TrainConfig(**_load_toml(os.path.join(exp_dir, 'train_cfg.dump.toml')))
    label = f'{train_cfg.arch} - {num_workers} workers{" - amp" if train_cfg.use_amp else ""}'
    test_log['label'] = label
    return test_log, test_log['val_acc1'].to_list()[-1], test_log['val_acc5'].to_list()[-1], test_log['time'].to_list()[-1]


def plot(logs: pd.DataFrame, x: str, y: str, img_dir: str, xscale: str='linear', yscale: str='linear'):
    plt.clf()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=logs, x=x, y=y, style='label', hue='label', errorbar=('sd', 2))
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(img_dir, dpi=300, bbox_inches='tight')



def main():
    exp_dirs = sys.argv[1:]

    time_values: Dict[str, list] = {}
    last_val_acc1_values: Dict[str, list] = {}
    last_val_acc5_values: Dict[str, list] = {}
    total_train_time_values: Dict[str, list] = {}
    for exp_dir in exp_dirs:
        train_log, last_val_acc1, last_val_acc5, total_train_time = read_train_log(exp_dir)
        for label in train_log['label'].unique():
            if label not in time_values:
                time_values[label] = []
                last_val_acc1_values[label] = []
                last_val_acc5_values[label] = []
                total_train_time_values[label] = []
            time_values[label].extend(train_log[train_log['label'] == label]['time'].unique().tolist())
            last_val_acc1_values[label].append(last_val_acc1)
            last_val_acc5_values[label].append(last_val_acc5)
            total_train_time_values[label].append(total_train_time)
    for label in time_values:
        time_values[label] = sorted(list(set(time_values[label])))


    interp_logs = []
    for exp_dir in exp_dirs:
        train_log, _, _, _ = read_train_log(exp_dir)
        interpolated = pd.DataFrame(columns=train_log.columns)

        label = train_log['label'].unique()[0]
        x = train_log['time'].to_list()
        y_val_acc1 = train_log['val_acc1'].to_list()
        y_val_acc5 = train_log['val_acc5'].to_list()

        f_val_acc1 = interp1d(x, y_val_acc1, kind='linear', fill_value='extrapolate') # type: ignore
        f_val_acc5 = interp1d(x, y_val_acc5, kind='linear', fill_value='extrapolate') # type: ignore

        interp_val_acc1 = f_val_acc1(time_values[label])
        interp_val_acc5 = f_val_acc5(time_values[label])

        interpolated['time'] = time_values[label]
        interpolated['val_acc1'] = interp_val_acc1
        interpolated['val_acc5'] = interp_val_acc5
        interpolated['label'] = label
        interp_logs.append(interpolated)
    
    interp_log = pd.concat(interp_logs)
    train_log = pd.concat([read_train_log(exp_dir)[0] for exp_dir in exp_dirs])

    os.makedirs('image', exist_ok=True)
    plot(interp_log, 'time', 'val_acc1', 'image/time_vs_acc1.png')
    plot(train_log, 'step', 'val_acc1', 'image/step_vs_acc1.png')
    plot(interp_log, 'time', 'val_acc5', 'image/time_vs_acc5.png')
    plot(train_log, 'step', 'val_acc5', 'image/step_vs_acc5.png')

    logger.info(f'Plots generated at image/ directory')

    for label in last_val_acc1_values:
        # log the last bleu score with 2x standard deviation
        _val_acc1 = np.array(last_val_acc1_values[label])
        _val_acc5 = np.array(last_val_acc5_values[label])
        _train_time = np.array(total_train_time_values[label]) / 60 / 60
        logger.info(f'{label} - last val acc1: {np.mean(_val_acc1):.4f} ± {2 * np.std(_val_acc1):.4f}')
        logger.info(f'{label} - last val acc5: {np.mean(_val_acc5):.4f} ± {2 * np.std(_val_acc5):.4f}')
        logger.info(f'{label} - total training time: {np.mean(_train_time):.4f} ± {2 * np.std(_train_time):.4f} hours')


if __name__ == '__main__':
    main()