#!/bin/bash
cd ..
DATA_FOLDER=/media/dsvitov/DATA/
OUTPUT_FOLDER=/media/dsvitov/DATA/output/Ours

# Process Tanks&Temples
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Train_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Train --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Training/Truck_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Truck --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Francis_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Francis --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Horse_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Horse --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Lighthouse_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Lighthouse --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval

# Mip-NeRF-360
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bonsai --model_path=${OUTPUT_FOLDER}/MipNerf/Bonsai --cap_max=160_000 --max_read_points=150_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/counter --model_path=${OUTPUT_FOLDER}/MipNerf/Counter --cap_max=160_000 --max_read_points=150_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/kitchen --model_path=${OUTPUT_FOLDER}/MipNerf/Kitchen --cap_max=160_000 --max_read_points=150_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/room --model_path=${OUTPUT_FOLDER}/MipNerf/Room --cap_max=160_000 --max_read_points=150_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bicycle --model_path=${OUTPUT_FOLDER}/MipNerf/Bicycle --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/stump --model_path=${OUTPUT_FOLDER}/MipNerf/Stump --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval
python train.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/garden --model_path=${OUTPUT_FOLDER}/MipNerf/Garden --cap_max=300_000 --max_read_points=290_000 --add_sky_box --eval

# Process DTU
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan24 --model_path=${OUTPUT_FOLDER}/DTU/scan24 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan37 --model_path=${OUTPUT_FOLDER}/DTU/scan37 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan40 --model_path=${OUTPUT_FOLDER}/DTU/scan40 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan55 --model_path=${OUTPUT_FOLDER}/DTU/scan55 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan63 --model_path=${OUTPUT_FOLDER}/DTU/scan63 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan65 --model_path=${OUTPUT_FOLDER}/DTU/scan65 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan69 --model_path=${OUTPUT_FOLDER}/DTU/scan69 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan83 --model_path=${OUTPUT_FOLDER}/DTU/scan83 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan97 --model_path=${OUTPUT_FOLDER}/DTU/scan97 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan105 --model_path=${OUTPUT_FOLDER}/DTU/scan105 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan106 --model_path=${OUTPUT_FOLDER}/DTU/scan106 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan110 --model_path=${OUTPUT_FOLDER}/DTU/scan110 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan114 --model_path=${OUTPUT_FOLDER}/DTU/scan114 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan118 --model_path=${OUTPUT_FOLDER}/DTU/scan118 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
python train.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan122 --model_path=${OUTPUT_FOLDER}/DTU/scan122 --cap_max=60_000 --max_read_points=60_000 --lambda_normal=0.05 --lambda_dist 100 --eval
