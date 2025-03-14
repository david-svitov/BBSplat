import os
import json
from collections import defaultdict
from glob import glob
from statistics import mean
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Script to averaging metrics values for the dataset')
    parser.add_argument('-f', '--folder', help='Path to the target folder', type=str, required=True)
    args = parser.parse_args()

    #======================================================
    results = glob(os.path.join(args.folder, "*/results.json"))

    metrics_statistic = defaultdict(list)
    for path in results:
        with open(path, "r") as f:
            metrics = json.load(f)
            metrics_statistic["PSNR"].append(metrics["ours_32000"]["PSNR"])
            metrics_statistic["SSIM"].append(metrics["ours_32000"]["SSIM"])
            metrics_statistic["LPIPS"].append(metrics["ours_32000"]["LPIPS"])

    print("PSNR:", mean(metrics_statistic["PSNR"]))
    print("SSIM:", mean(metrics_statistic["SSIM"]))
    print("LPIPS:", mean(metrics_statistic["LPIPS"]))

    #======================================================
    results = glob(os.path.join(args.folder, "*/speed.json"))

    metrics_statistic = defaultdict(list)
    for path in results:
        with open(path, "r") as f:
            metrics = json.load(f)
            metrics_statistic["points"].append(metrics["points"])
            metrics_statistic["train_time"].append(metrics["train_time"])
            metrics_statistic["train_time_std"].append(metrics["train_time_std"])

    print("Points:", mean(metrics_statistic["points"]))
    print("FPS:", mean(metrics_statistic["train_time"]), "±", mean(metrics_statistic["train_time_std"]))

    #======================================================
    files_1 = glob(os.path.join(args.folder, "*/point_cloud/iteration_32000/texture_color.npz"))
    files_2 = glob(os.path.join(args.folder, "*/point_cloud/iteration_32000/texture_alpha.npz"))
    files_3 = glob(os.path.join(args.folder, "*/point_cloud/iteration_32000/point_cloud.ply"))

    size_statistic = []
    for f1, f2, f3 in zip(files_1, files_2, files_3):
        total_size = 0
        file_stats = os.stat(f1)
        total_size += file_stats.st_size / (1024 * 1024)
        file_stats = os.stat(f2)
        total_size += file_stats.st_size / (1024 * 1024)
        file_stats = os.stat(f3)
        total_size += file_stats.st_size / (1024 * 1024)
        size_statistic.append(total_size)

    print("Size:", mean(size_statistic), " MB")
