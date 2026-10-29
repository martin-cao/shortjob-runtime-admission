import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from camera_ready_data import LABELS, read_json, require


def render(results, output):
    require(not output.exists(), '图像输出目录必须尚不存在')
    values = read_json(results / 'plotted_values.json')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    devices = list(LABELS)
    counts = np.array(values['fig_oracle_action_mix']['counts'])
    labels = ['Fixed inference', 'CV inference', 'LLM decode', 'Decode toy', 'Prefill toy',
              'Recommendation', 'RL policy', 'Kernel chain']
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout='constrained')
    cmap = ListedColormap(['#f4f8f6', '#d7e9e1', '#a4cbb9', '#559e80', '#007354'])
    for j, (device, ax) in enumerate(zip(devices, axes.flat, strict=True)):
        panel = counts[:, j * 4:j * 4 + 4]
        ax.imshow(panel, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 5), 5), aspect='auto')
        ax.set_title(LABELS[device])
        ax.set_xticks(range(4), ['10', '50', '100', '500'])
        ax.set_yticks(range(8), labels)
        ax.set_xlabel('Job length N (steps)')
        for row in range(8):
            for column in range(4):
                count = panel[row, column]
                ax.text(column, row, str(count), ha='center', va='center',
                        color='white' if count >= 3 else '#23352d')
    fig.suptitle('Graph oracle count across four batch sizes')
    for suffix in ['pdf', 'png']:
        fig.savefig(output / f'graph_boundaries.{suffix}', dpi=600)
    plt.close(fig)
    points = [p for p in values['real_model_results'] if p['steps'] == 500]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True, layout='constrained')
    models = [('mobilenet_v3_large', 'MobileNetV3-Large'), ('resnet50', 'ResNet-50'), ('vit_b_16', 'ViT-B/16')]
    for ax, (model, title) in zip(axes, models, strict=True):
        for i, device in enumerate(devices):
            p = next(p for p in points if p['model'] == model and p['device'] == device)
            ax.errorbar(p['median'], i, xerr=[[p['median'] - p['low']], [p['high'] - p['median']]],
                        fmt='o', color='#0072B2', capsize=3)
        ax.set_title(title)
        ax.set_yticks(range(4), list(LABELS.values()))
        ax.set_ylim(3.5, -.5)
        ax.axvline(0, color='#646B70', linestyle='--', linewidth=.8)
        ax.set_xlabel('Change in total service time (%)')
    fig.suptitle('N = 500; median block effect and observed min/max envelope')
    for suffix in ['pdf', 'png']:
        fig.savefig(output / f'real_model_effects.{suffix}', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='将已重建数值绘制到新的独立图像目录')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.results.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
