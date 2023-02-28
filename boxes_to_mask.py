import matplotlib
import json
import argparse
import numpy as np

#These are hardcoded in as they are not included in the project JSON and they are constant for all images.
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 2448

def create_map(regions):
    pass

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
        create_map(regions)