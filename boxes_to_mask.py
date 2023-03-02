from matplotlib.path import Path
import json
import argparse
import numpy as np
import cv2
import concurrent.futures
import os

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

def create_mask_for_image(image,path):
    filename = image["filename"][:-4] #Remove extension
    output_file = f"{path}/{filename}_mask.png"
    #Check if file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists")
        return
    
    print(f"Creating mask for: {filename}")
    regions = extract_regions_from_json(image)
    paths = regions_to_paths(regions)
    mask = create_mask(paths)
    cv2.imwrite(output_file,mask)
    print(f"Mask created and written to {output_file}")

def extract_regions_from_json(image_json):
    regions = {}
    for region in image_json["regions"]: 
        label = region["region_attributes"]["class"]
        xs =region["shape_attributes"]["all_points_x"]
        ys = region["shape_attributes"]["all_points_y"]
        points = np.array([[x,y] for x,y in zip(xs,ys)])
        if label in regions:
            regions[label].append(points)
        else:
            regions[label] = [points]
    return regions

def create_mask(paths):
    mask = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.int64)
    for y in range(IMAGE_HEIGHT):
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
    parser.add_argument('file_path')
    parser.add_argument('-t','--max-threads',type=int,default=1)
    args = parser.parse_args()

    f = open(args.json_file)
    project = json.load(f)
    f.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as thread_pool:
        thread_pool.map(lambda x:create_mask_for_image(x,args.file_path),project["_via_img_metadata"].values())