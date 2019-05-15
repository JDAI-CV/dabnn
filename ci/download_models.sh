#! /usr/bin/env bash

wget "https://drive.google.com/uc?export=download&id=1FYqF5BvYO2kl13bn28sheKtgverkbbQN" -O model_imagenet.dab 
adb push model_imagenet.dab /data/local/tmp 
wget "https://drive.google.com/uc?export=download&id=1frtRL1O0zhtJvPFbhE8COFl1-WWHIaOW" -O model_imagenet_stem.dab 
adb push model_imagenet_stem.dab /data/local/tmp/ 
