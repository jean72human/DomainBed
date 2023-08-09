#!/bin/bash

# This script is used to run ERM then JTT

python -m domainbed.scripts.train_n --dataset SpawriousM2M_easy --algorithm ERM --data_dir ./data/spawrious224 --output_dir ./ERM_JTT/SpawriousM2M_easy/ERM/ --steps 2000


python -m domainbed.scripts.train_n --dataset SpawriousM2M_easy_JTT --algorithm JTT --data_dir ./data/spawrious224 --pretrained_model_path ./ERM_JTT/SpawriousM2M_easy/ERM/model_step20.pkl --step 50000 --output_dir ./ERM_JTT/SpawriousM2M_easy/JTT/ --upweight 1.0 