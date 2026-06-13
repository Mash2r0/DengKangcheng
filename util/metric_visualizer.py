import csv
import math
import os
import time
from collections import defaultdict


class MetricsVisualizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.root = os.path.join(opt.checkpoints_dir, opt.name, 'metric_plots')
        self.csv_path = os.path.join(self.root, 'metrics_long.csv')
        self.plot_freq = max(int(getattr(opt, 'metric_plot_freq', 1)), 1)
        self._warned = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    @staticmethod
    def _sanitize(name):
        return str(name).replace('\\', '_').replace('/', '_').replace(' ', '_')

    @staticmethod
    def _meter_items(meters):
        if hasattr(meters, 'keys'):
            return [(key, float(meters[key])) for key in sorted(meters.keys())]
        return [(key, float(value)) for key, value in sorted(meters.items())]

    def record(self, scope, epoch, iterations, meters):
        items = self._meter_items(meters)
        if not items:
            return

        exists = os.path.exists(self.csv_path)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'scope', 'epoch', 'iterations', 'metric', 'value'])
            if not exists:
                writer.writeheader()
            for metric, value in items:
                writer.writerow({
                    'timestamp': timestamp,
                    'scope': scope,
                    'epoch': epoch,
                    'iterations': iterations,
                    'metric': metric,
                    'value': value,
                })

        if epoch % self.plot_freq == 0:
            self._plot_scope(scope)

    def _read_scope(self, scope):
        data = defaultdict(list)
        if not os.path.exists(self.csv_path):
            return data

        with open(self.csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['scope'] != scope:
                    continue
                metric = row['metric']
                x = float(row['epoch'])
                value = float(row['value'])
                data[metric].append((x, value))
        return data

    def _plot_scope(self, scope):
        data = self._read_scope(scope)
        if not data:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as err:
            if not self._warned:
                print('[w] metric plot disabled: %s' % err)
                self._warned = True
            return

        metrics = sorted(data.keys())
        ncols = min(3, len(metrics))
        nrows = int(math.ceil(len(metrics) / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // ncols][idx % ncols]
            points = data[metric]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker='o', linewidth=1.5, markersize=3)
            ax.set_title(metric)
            ax.set_xlabel('epoch')
            ax.grid(True, alpha=0.3)

        for idx in range(len(metrics), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis('off')

        fig.suptitle(scope)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        path = os.path.join(self.root, '%s.png' % self._sanitize(scope))
        fig.savefig(path, dpi=150)
        plt.close(fig)
