#!/bin/bash
cd ..
DATA_FOLDER=/media/dsvitov/DATA/
OUTPUT_FOLDER=/media/dsvitov/DATA/output/Ours

# Process Tanks&Temples
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Train_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Train --skip_mesh
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Training/Truck_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Truck --skip_mesh
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Francis_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Francis --skip_mesh
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Horse_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Horse --skip_mesh
python render.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Lighthouse_COLMAP_big --model_path=${OUTPUT_FOLDER}/TnT/Lighthouse --skip_mesh

# Mip-NeRF-360
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bonsai --model_path=${OUTPUT_FOLDER}/MipNerf/Bonsai --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/counter --model_path=${OUTPUT_FOLDER}/MipNerf/Counter --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/kitchen --model_path=${OUTPUT_FOLDER}/MipNerf/Kitchen --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/room --model_path=${OUTPUT_FOLDER}/MipNerf/Room --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bicycle --model_path=${OUTPUT_FOLDER}/MipNerf/Bicycle --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/stump --model_path=${OUTPUT_FOLDER}/MipNerf/Stump --skip_mesh
python render.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/garden --model_path=${OUTPUT_FOLDER}/MipNerf/Garden --skip_mesh

# Process DTU
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan24 --model_path=${OUTPUT_FOLDER}/DTU/scan24 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan37 --model_path=${OUTPUT_FOLDER}/DTU/scan37 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan40 --model_path=${OUTPUT_FOLDER}/DTU/scan40 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan55 --model_path=${OUTPUT_FOLDER}/DTU/scan55 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan63 --model_path=${OUTPUT_FOLDER}/DTU/scan63 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan65 --model_path=${OUTPUT_FOLDER}/DTU/scan65 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan69 --model_path=${OUTPUT_FOLDER}/DTU/scan69 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan83 --model_path=${OUTPUT_FOLDER}/DTU/scan83 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan97 --model_path=${OUTPUT_FOLDER}/DTU/scan97 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan105 --model_path=${OUTPUT_FOLDER}/DTU/scan105 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan106 --model_path=${OUTPUT_FOLDER}/DTU/scan106 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan110 --model_path=${OUTPUT_FOLDER}/DTU/scan110 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan114 --model_path=${OUTPUT_FOLDER}/DTU/scan114 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan118 --model_path=${OUTPUT_FOLDER}/DTU/scan118 --skip_mesh
python render.py -s ${DATA_FOLDER}DTU/dtu/DTU/scan122 --model_path=${OUTPUT_FOLDER}/DTU/scan122 --skip_mesh
