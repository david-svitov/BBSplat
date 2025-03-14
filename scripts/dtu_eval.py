import json
import os
from argparse import ArgumentParser
from glob import glob
from statistics import mean

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

points = {
'scan24': 30_000,
'scan37': 30_000,
'scan40': 30_000,
'scan55': 60_000,
'scan63': 60_000,
'scan65': 60_000,
'scan69': 60_000,
'scan83': 60_000,
'scan97': 60_000,
'scan105': 30_000,
'scan106': 60_000,
'scan110': 60_000,
'scan114': 60_000,
'scan118': 60_000,
'scan122': 60_000,
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    for scene in dtu_scenes:
        common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000 --lambda_normal=0.05 --cap_max=" + str(points[scene]) + " --max_read_points=" + str(points[scene])
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 32000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 32000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_32000/"
        iteration = 32000
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_32000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {script_dir}/tmp/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)

    results = glob(f"{script_dir}/tmp/*/results.json")
    overall = []
    for path in results:
        with open(path, "r") as f:
            metrics = json.load(f)
            overall.append(metrics["overall"])

    print("Mean CD:", mean(overall))