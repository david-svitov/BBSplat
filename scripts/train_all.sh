#!/bin/bash
cd ..
DATA_FOLDER=/media/dsvitov/DATA/
OUTPUT_FOLDER=/media/dsvitov/DATA/output/debug

# Process Tanks&Temples
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Train_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Train --cap_max=160_000 --max_read_points=150_000 --add_sky_box
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Training/Truck_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Truck --cap_max=160_000 --max_read_points=150_000 --add_sky_box
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Francis_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Francis --cap_max=160_000 --max_read_points=150_000 --add_sky_box
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Horse_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Horse --cap_max=160_000 --max_read_points=150_000 --add_sky_box
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Lighthouse_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Lighthouse --cap_max=160_000 --max_read_points=150_000 --add_sky_box

# Mip-NeRF-360
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bonsai/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Bonsai --cap_max=160_000 --max_read_points=160_000
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/counter/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Counter --cap_max=160_000 --max_read_points=160_000
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/kitchen/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Kitchen --cap_max=160_000 --max_read_points=160_000
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/room/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Room --cap_max=160_000 --max_read_points=160_000

# Process DTU
python train.py -s ${DATA_FOLDER}/DTU/selected/scan24_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan24 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan37_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan37 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan40_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan40 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan55_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan55 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan63_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan63 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan65_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan65 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan69_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan69 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan83_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan83 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan97_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan97 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan105_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan105 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan106_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan106 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan110_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan110 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan114_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan114 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan118_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan118 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05
python train.py -s ${DATA_FOLDER}/DTU/selected/scan122_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan122 --cap_max=30_000 --max_read_points=30_000 --lambda_dist=100.0 --lambda_normal=0.05

