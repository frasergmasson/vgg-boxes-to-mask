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
    "growler": [0,0,255],
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
    regions,boxes  = extract_regions_from_json(image)
    #Quit out if there are no regions in image: impossible to make a mask
    if len(regions) == 0:
        return
    
    paths = regions_to_paths(regions)
    mask = create_mask(paths,boxes)
    cv2.imwrite(output_file,mask)
    print(f"Mask created and written to {output_file}")

def extract_regions_from_json(image_json):
    regions = {}
    boxes = {}
    for region in image_json["regions"]: 
        label = region["region_attributes"]["class"]
        xs =region["shape_attributes"]["all_points_x"]
        ys = region["shape_attributes"]["all_points_y"]
        points = np.array([[x,y] for x,y in zip(xs,ys)])
        box = (min(xs)-1,min(ys)-1,max(xs)+1,max(ys)+1)
        if label in regions:
            regions[label].append(points)
            boxes[label].append(box)
        else:
            regions[label] = [points]
            boxes[label] = [box]
    return regions,boxes

def create_mask(paths,boxes):
    mask = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.int64)
    for label in paths.keys():
        mask = fill_in(mask,label_colours[label],paths[label],boxes[label])
    return mask

def fill_in(mask,colour,paths,boxes):
    for i,path in enumerate(paths):
        minx,miny,maxx,maxy = boxes[i]
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                if path.contains_point((x,y)):
                    mask[y,x,:] = colour
    return mask

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
    annotations = json.load(f)
    f.close()

    # for a in annotations.values():
    #     create_mask_for_image(a,args.file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as thread_pool:
        thread_pool.map(lambda x:create_mask_for_image(x,args.file_path),annotations.values())