#!/bin/bash
#1 original images
#2 annotated images
#3 resized images
#4 masks
#5 json annotations
#6 height
#7 width
#8 threads
python3 copy_annotated_images.py $1 $2 $5
python3 resize_images.py $2 $3 $6 $7 -t $8
python3 boxes_to_mask.py $5 $4 -i $3 -g -t $8