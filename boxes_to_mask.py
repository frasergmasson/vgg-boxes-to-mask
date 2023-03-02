from matplotlib.path import Path
import json
import argparse
import numpy as np
import cv2

#These are hardcoded in as they are not included in the project JSON and they are constant for all images.
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 2448

label_order = [
    "growlers",
    "iceberg",
    "glacier"
]

label_colours = {
    "glacier": [255,0,0],
    "iceberg": [0,255,0],
    "growlers": [0,0,255],
    "0": [0,0,0]
}

def create_mask(paths):
    mask = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.int64)
    for y in range(IMAGE_HEIGHT):
        print(y)
        for x in range(IMAGE_WIDTH):
            mask[y,x,:] = label_colours[assign_point_label((x,y),paths)]
    return mask

def assign_point_label(point,paths):
    for label in label_order:
        if label in paths:
            label_paths = paths[label]
            if point_in_paths(point,label_paths):
                return label
    return "0"

def point_in_paths(point,paths):
    for path in paths:
        if path.contains_point(point):
            return True
    return False

def regions_to_paths(regions):
    paths = {}
    for label in regions.keys():
        paths[label] = [Path(region) for region in regions[label]]
    return paths

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    f = open(args.json_file)
    project = json.load(f)
    f.close()

    for image in project["_via_img_metadata"].values():
        regions = {}
        for region in image["regions"]: 
            label = region["region_attributes"]["class"]
            xs =region["shape_attributes"]["all_points_x"]
            ys = region["shape_attributes"]["all_points_y"]
            points = np.array([[x,y] for x,y in zip(xs,ys)])
            if label in regions:
                regions[label].append(points)
            else:
                regions[label] = [points]
        paths = regions_to_paths(regions)

        cv2.imwrite("mask.png",create_mask(paths))