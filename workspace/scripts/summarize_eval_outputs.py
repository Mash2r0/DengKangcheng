import argparse
import csv
import os
import sys
from glob import glob
from os.path import join

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = os.path.abspath(join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


EXPECTED_COUNTS = {
    "ceilnet_table2": 100,
    "CEILNet_table2": 100,
    "real20": 20,
    "objects": 200,
    "postcard": 179,
    "wild": 101,
    "sir2_withgt": 480,
}

CANONICAL_DATASET = {
    "CEILNet_table2": "ceilnet_table2",
    "ceilnet_table2": "ceilnet_table2",
}


def canonical_dataset(name):
    return CANONICAL_DATASET.get(name, name)


def read_image(path):
    return np.array(Image.open(path).convert("RGB"))


def bandwise_psnr(correct, estimate):
    return np.mean(
        [
            peak_signal_noise_ratio(correct[..., c], estimate[..., c], data_range=255)
            for c in range(correct.shape[-1])
        ]
    )


def bandwise_ssim(correct, estimate):
    return np.mean(
        [
            structural_similarity(correct[..., c], estimate[..., c], data_range=255)
            for c in range(correct.shape[-1])
        ]
    )


def compare_ncc(correct, estimate):
    correct = correct.astype(np.float64)
    estimate = estimate.astype(np.float64)
    return np.mean((correct - np.mean(correct)) * (estimate - np.mean(estimate))) / (
        np.std(correct) * np.std(estimate)
    )


def local_error_fast(correct, estimate, window_size=20, window_shift=10):
    correct = correct.astype(np.float64)
    estimate = estimate.astype(np.float64)
    total = 0.0
    ssq = 0.0
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:
        sliding_window_view = None

    if sliding_window_view is None:
        # Compatibility fallback; this is slower but follows util.index.local_error.
        for c in range(correct.shape[-1]):
            for i in range(0, correct.shape[0] - window_size + 1, window_shift):
                for j in range(0, correct.shape[1] - window_size + 1, window_shift):
                    cw = correct[i : i + window_size, j : j + window_size, c]
                    ew = estimate[i : i + window_size, j : j + window_size, c]
                    denom = np.sum(ew**2)
                    alpha = np.sum(cw * ew) / denom if denom > 1e-5 else 0.0
                    ssq += np.sum((cw - alpha * ew) ** 2)
                    total += np.sum(cw**2)
        return ssq / total

    for c in range(correct.shape[-1]):
        cw = sliding_window_view(correct[..., c], (window_size, window_size))[::window_shift, ::window_shift]
        ew = sliding_window_view(estimate[..., c], (window_size, window_size))[::window_shift, ::window_shift]
        denom = np.sum(ew * ew, axis=(-1, -2))
        numer = np.sum(cw * ew, axis=(-1, -2))
        alpha = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-5)
        ssq += np.sum((cw - alpha[..., None, None] * ew) ** 2)
        total += np.sum(cw**2)
    return ssq / total


def quality_assess_fast(estimate, correct):
    return {
        "PSNR": float(bandwise_psnr(correct, estimate)),
        "SSIM": float(bandwise_ssim(correct, estimate)),
        "LMSE": float(local_error_fast(correct, estimate, 20, 10)),
        "NCC": float(compare_ncc(correct, estimate)),
    }


def find_prediction(sample_dir, checkpoint):
    exact = join(sample_dir, f"errnet_r3lite_{checkpoint}.png")
    if os.path.exists(exact):
        return exact
    candidates = [
        p
        for p in glob(join(sample_dir, "*.png"))
        if os.path.basename(p) not in {"m_input.png", "t_label.png"}
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def summarize_dataset(dataset_dir, checkpoint, dataset_name):
    rows = []
    for sample_dir in sorted(glob(join(dataset_dir, "*"))):
        if not os.path.isdir(sample_dir):
            continue
        target_path = join(sample_dir, "t_label.png")
        pred_path = find_prediction(sample_dir, checkpoint)
        if not os.path.exists(target_path) or pred_path is None:
            continue
        pred = read_image(pred_path)
        target = read_image(target_path)
        h = min(pred.shape[0], target.shape[0])
        w = min(pred.shape[1], target.shape[1])
        rows.append(quality_assess_fast(pred[:h, :w], target[:h, :w]))

    if not rows:
        return None

    metrics = {}
    for key in ["LMSE", "NCC", "PSNR", "SSIM"]:
        metrics[key] = float(sum(row[key] for row in rows) / len(rows))
    sample_count = len(rows)
    expected = EXPECTED_COUNTS.get(dataset_name, sample_count)
    metrics["sample_count"] = sample_count
    metrics["expected_count"] = expected
    metrics["complete"] = sample_count == expected
    return metrics


def write_csv(path, rows):
    fields = [
        "checkpoint",
        "dataset",
        "sample_count",
        "expected_count",
        "complete",
        "LMSE",
        "NCC",
        "PSNR",
        "SSIM",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path, row):
    fields = [
        "checkpoint",
        "dataset",
        "sample_count",
        "expected_count",
        "complete",
        "LMSE",
        "NCC",
        "PSNR",
        "SSIM",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown(path, rows, complete_only=False):
    with open(path, "w", encoding="utf-8") as f:
        title = "Complete" if complete_only else "All"
        f.write(f"# {title} Eval Output Summary\n\n")
        f.write("| checkpoint | dataset | n | complete | LMSE | NCC | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if complete_only and not row["complete"]:
                continue
            f.write(
                "| {checkpoint} | {dataset} | {sample_count}/{expected_count} | {complete} | {LMSE:.4f} | {NCC:.4f} | {PSNR:.4f} | {SSIM:.4f} |\n".format(
                    **row
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="workspace/results/checkpoint_sweep_outputs")
    parser.add_argument("--output_dir", default="workspace/results/checkpoint_sweep")
    args = parser.parse_args()

    rows = []
    csv_path = join(args.output_dir, "checkpoint_sweep_from_outputs.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    for checkpoint_dir in sorted(glob(join(args.image_dir, "*"))):
        if not os.path.isdir(checkpoint_dir):
            continue
        checkpoint = os.path.basename(checkpoint_dir)
        for dataset_dir in sorted(glob(join(checkpoint_dir, "*"))):
            if not os.path.isdir(dataset_dir):
                continue
            raw_dataset = os.path.basename(dataset_dir)
            dataset = canonical_dataset(raw_dataset)
            summary = summarize_dataset(dataset_dir, checkpoint, raw_dataset)
            if summary is None:
                continue
            row = {"checkpoint": checkpoint, "dataset": dataset}
            row.update(summary)
            rows.append(row)
            append_csv(csv_path, row)
            print(
                "[summary] {checkpoint} {dataset} {sample_count}/{expected_count} LMSE={LMSE:.4f} NCC={NCC:.4f} PSNR={PSNR:.4f} SSIM={SSIM:.4f}".format(
                    **row
                )
            )

    write_csv(join(args.output_dir, "checkpoint_sweep_from_outputs.csv"), rows)
    write_markdown(join(args.output_dir, "checkpoint_sweep_from_outputs.md"), rows)
    write_markdown(join(args.output_dir, "checkpoint_sweep_complete.md"), rows, complete_only=True)


if __name__ == "__main__":
    main()
