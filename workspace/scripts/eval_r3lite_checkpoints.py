import argparse
import csv
import os
import sys
from os.path import join

REPO_ROOT = os.path.abspath(join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options import TrainOptions
from test_errnet import EVAL_DATASETS, build_eval_dataloader
import util.util as util


DEFAULT_DATASETS = [
    "ceilnet_table2",
    "real20",
    "objects",
    "postcard",
    "wild",
    "sir2_withgt",
]


DEFAULT_CHECKPOINTS = [
    ("aligned55", "checkpoints/errnet_r3lite_scratch/errnet_055_00425260.pt"),
    ("aligned60", "checkpoints/errnet_r3lite_scratch/errnet_060_00463920.pt"),
    ("unaligned75", "checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_075_00583650.pt"),
    ("unaligned80", "checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt"),
]


def quiet_progress_bar(current, total, msg=None):
    if current == total - 1:
        print(msg or "")


def make_opt(name, checkpoint_path, nthreads):
    argv = [
        sys.argv[0],
        "--name",
        name,
        "-r",
        "--icnn_path",
        checkpoint_path,
        "--hyper",
        "--inet",
        "errnet_r3lite",
        "--display_id",
        "0",
        "--nThreads",
        str(nthreads),
        "--no-verbose",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        parser = TrainOptions()
        parser.isTrain = False
        opt = parser.parse()
    finally:
        sys.argv = old_argv

    opt.isTrain = False
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False
    return opt


def write_csv(path, rows):
    fieldnames = ["checkpoint", "dataset", "LMSE", "NCC", "PSNR", "SSIM"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path, row):
    fieldnames = ["checkpoint", "dataset", "LMSE", "NCC", "PSNR", "SSIM"]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ERRNet-R3Lite Checkpoint Sweep\n\n")
        f.write("| checkpoint | dataset | LMSE | NCC | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {checkpoint} | {dataset} | {LMSE:.4f} | {NCC:.4f} | {PSNR:.4f} | {SSIM:.4f} |\n".format(
                    **row
                )
            )


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--output_dir", default="workspace/results/checkpoint_sweep")
    argp.add_argument("--image_dir", default="workspace/results/checkpoint_sweep_outputs")
    argp.add_argument("--data_root", default="./datasets/processed_data")
    argp.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=sorted(EVAL_DATASETS))
    argp.add_argument("--checkpoint_labels", nargs="+", default=None)
    argp.add_argument(
        "--extra_checkpoint",
        action="append",
        default=[],
        help="extra checkpoint in label=path form; can be repeated",
    )
    argp.add_argument("--nThreads", type=int, default=0)
    argp.add_argument("--no_save_images", action="store_true")
    args = argp.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)
    csv_path = join(args.output_dir, "checkpoint_sweep.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    util.progress_bar = quiet_progress_bar
    datasets.util.progress_bar = quiet_progress_bar if hasattr(datasets, "util") else quiet_progress_bar

    rows = []
    cudnn.benchmark = True

    checkpoints = list(DEFAULT_CHECKPOINTS)
    for item in args.extra_checkpoint:
        if "=" not in item:
            raise ValueError(f"--extra_checkpoint must use label=path form: {item}")
        label, path = item.split("=", 1)
        checkpoints.append((label, path))

    for label, checkpoint_path in checkpoints:
        if args.checkpoint_labels and label not in args.checkpoint_labels:
            continue
        if not os.path.exists(checkpoint_path):
            print(f"[skip] {label}: {checkpoint_path}")
            continue

        opt = make_opt(f"errnet_r3lite_{label}", checkpoint_path, args.nThreads)
        engine = Engine(opt)

        for dataset_key in args.datasets:
            print(f"[eval] {label} / {dataset_key}")
            spec, dataloader = build_eval_dataloader(opt, args.data_root, dataset_key)
            savedir = None if args.no_save_images else join(args.image_dir, label, spec["save_subdir"])
            meters = engine.eval(dataloader, dataset_name=spec["dataset_name"], savedir=savedir)
            row = {
                "checkpoint": label,
                "dataset": dataset_key,
                "LMSE": float(meters["LMSE"]),
                "NCC": float(meters["NCC"]),
                "PSNR": float(meters["PSNR"]),
                "SSIM": float(meters["SSIM"]),
            }
            rows.append(row)
            append_csv(csv_path, row)
            print(
                "[metric] {checkpoint} {dataset} LMSE={LMSE:.4f} NCC={NCC:.4f} PSNR={PSNR:.4f} SSIM={SSIM:.4f}".format(
                    **row
                )
            )

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(csv_path, rows)
    write_markdown(join(args.output_dir, "checkpoint_sweep.md"), rows)
    print(f"[done] wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
