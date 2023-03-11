from matplotlib.path import Path
import json
import argparse
import numpy as np
import cv2
import concurrent.futures
import os

IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 2448

def extract_regions_from_json(image):
    regions = []
    boxes = []
    for region in image["regions"]: 
        xs =region["shape_attributes"]["all_points_x"]
        ys = region["shape_attributes"]["all_points_y"]
        points = np.array([[x,y] for x,y in zip(xs,ys)])
        regions.append(points)
        boxes.append((min(xs)-1,min(ys)-1,max(xs)+1,max(ys)+1))
    return regions,boxes

#Returns a NxHxW array where N is the number of instances
def create_masks(paths,boxes):
    masks = np.zeros((len(paths),IMAGE_HEIGHT,IMAGE_WIDTH),dtype=np.int64)
    for i,path in enumerate(paths):
        minx,miny,maxx,maxy = boxes[i]
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                if path.contains_point((x,y)):
                    masks[i,y,x] = 255
    return masks

def create_masks_for_image(image,path):
    try:
        filename = image["filename"][:-4] #Remove extension
        output_directory = f"{path}/{filename}"
        #Check if file already exists
        if os.path.exists(output_directory):
            print(f"Directory {output_directory} already exists")
            return
        
        print(f"Creating mask for: {filename}")
        regions,boxes = extract_regions_from_json(image)
        #Quit out if there are no regions in image: impossible to make a mask
        if len(regions) == 0:
            return
        
        paths = [Path(region) for region in regions]
        masks = create_masks(paths,boxes)
        os.mkdir(output_directory)
        for i,mask in enumerate(masks):
            output_file = f"{output_directory}/{i}.png"
            cv2.imwrite(output_file,mask)
        print(f"Masks created and written to {output_directory}")
    except Exception as e:
        print(e)


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
        thread_pool.map(lambda x:create_masks_for_image(x,args.file_path),project.values())