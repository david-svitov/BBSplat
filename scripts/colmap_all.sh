#!/bin/bash
cd ..
DATA_FOLDER=/media/dsvitov/DATA/

# Process Tanks&Temples
# First put images in /*_COLMAP/input folder
python3 convert.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Train_COLMAP/
python3 convert.py -s ${DATA_FOLDER}/Tanks_and_Temples/Training/Truck_COLMAP/
python3 convert.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Francis_COLMAP/
python3 convert.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Horse_COLMAP/
python3 convert.py -s ${DATA_FOLDER}/Tanks_and_Temples/Intermediate/Lighthouse_COLMAP/

# Mip-NeRF-360
# First put images in /*/COLMAP/input folder
python3 convert.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/bonsai/COLMAP
python3 convert.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/counter/COLMAP
python3 convert.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/kitchen/COLMAP
python3 convert.py -s ${DATA_FOLDER}/Mip-NeRF-360/360_v2/room/COLMAP

# Process DTU
# First put images in /*_COLMAP/input folder
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan24_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan37_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan40_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan55_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan63_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan65_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan69_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan83_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan97_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan105_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan106_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan110_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan114_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan118_COLMAP
python3 convert.py -s ${DATA_FOLDER}/DTU/selected/scan122_COLMAP

