#!/bin/bash
cd ..
DATA_FOLDER=/media/dsvitov/DATA/
OUTPUT_FOLDER=/media/dsvitov/DATA/output/debug

# Process Tanks&Temples
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Train_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Train
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Training/Truck_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Truck
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Francis_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Francis
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Horse_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Horse
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Lighthouse_COLMAP --model_path=${OUTPUT_FOLDER}/TnT/Lighthouse

# Mip-NeRF-360
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bonsai/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Bonsai
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/counter/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Counter
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/kitchen/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Kitchen
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/room/COLMAP --model_path=${OUTPUT_FOLDER}/MipNerf/Room

# Process DTU
python render.py -s ${DATA_FOLDER}/DTU/selected/scan24_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan24
python render.py -s ${DATA_FOLDER}/DTU/selected/scan37_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan37
python render.py -s ${DATA_FOLDER}/DTU/selected/scan40_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan40
python render.py -s ${DATA_FOLDER}/DTU/selected/scan55_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan55
python render.py -s ${DATA_FOLDER}/DTU/selected/scan63_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan63
python render.py -s ${DATA_FOLDER}/DTU/selected/scan65_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan65
python render.py -s ${DATA_FOLDER}/DTU/selected/scan69_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan69
python render.py -s ${DATA_FOLDER}/DTU/selected/scan83_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan83
python render.py -s ${DATA_FOLDER}/DTU/selected/scan97_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan97
python render.py -s ${DATA_FOLDER}/DTU/selected/scan105_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan105
python render.py -s ${DATA_FOLDER}/DTU/selected/scan106_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan106
python render.py -s ${DATA_FOLDER}/DTU/selected/scan110_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan110
python render.py -s ${DATA_FOLDER}/DTU/selected/scan114_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan114
python render.py -s ${DATA_FOLDER}/DTU/selected/scan118_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan118
python render.py -s ${DATA_FOLDER}/DTU/selected/scan122_COLMAP --model_path=${OUTPUT_FOLDER}/DTU/scan122

