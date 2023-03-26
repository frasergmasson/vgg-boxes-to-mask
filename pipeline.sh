#!/bin/bash
#1 original images
#2 annotated images
#3 resized images
#4 masks
#5 json annotations
#6 height
#7 width
#8 threads

#Make output directories
mkdir -p $2
mkdir -p $3
mkdir -p $4

#Clear output directories
rm $2/*
rm $3/*
rm $4/*

python3 copy_annotated_images.py $1 $2 $5
python3 resize_images.py $2 $3 $6 $7 -t $8
python3 boxes_to_mask.py $5 $4 -i $3 -g -t $8