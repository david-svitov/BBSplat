#!/bin/bash
cd ..
OUTPUT_FOLDER=/media/dsvitov/DATA/output/Ours

# Process Tanks&Temples
python metrics.py --model_path=${OUTPUT_FOLDER}/TnT/Train
python metrics.py --model_path=${OUTPUT_FOLDER}/TnT/Truck
python metrics.py --model_path=${OUTPUT_FOLDER}/TnT/Francis
python metrics.py --model_path=${OUTPUT_FOLDER}/TnT/Horse
python metrics.py --model_path=${OUTPUT_FOLDER}/TnT/Lighthouse

# Mip-NeRF-360
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Bonsai
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Counter
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Kitchen
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Room
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Bicycle
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Stump
python metrics.py --model_path=${OUTPUT_FOLDER}/MipNerf/Garden

# Process DTU
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan24
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan37
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan40
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan55
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan63
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan65
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan69
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan83
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan97
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan105
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan106
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan110
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan114
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan118
python metrics.py --model_path=${OUTPUT_FOLDER}/DTU/scan122

# Average metrics for each dataset
python scripts/average_error.py --folder ${OUTPUT_FOLDER}/TnT
python scripts/average_error.py --folder ${OUTPUT_FOLDER}/MipNerf
python scripts/average_error.py --folder ${OUTPUT_FOLDER}/DTU
