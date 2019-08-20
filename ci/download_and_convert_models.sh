#! /usr/bin/env bash

wget "https://drive.google.com/uc?export=download&id=1Xp3HB51H6Nhl6e555ieJubVutQake5sR" -O model_imagenet.onnx
./build_onnx2bnn/tools/onnx2bnn/onnx2bnn model_imagenet.onnx model_imagenet.dab --aggressive --verbose
adb push model_imagenet.dab /data/local/tmp 
wget "https://drive.google.com/uc?export=download&id=1zu48CFptAGZ91IDCBPJSPM0bxDuPm9HS" -O model_imagenet_stem.onnx
./build_onnx2bnn/tools/onnx2bnn/onnx2bnn model_imagenet_stem.onnx model_imagenet_stem.dab --aggressive --verbose
adb push model_imagenet_stem.dab /data/local/tmp/ 
