#!/bin/bash

cd /mnt/volume_lon1_01/Project/movs_classification_2023
echo "running hardstop script..."
docker/hardstop.sh &
wait
echo "running start_manual script..."
docker/start_manual.sh &
wait
echo "api reload completed"