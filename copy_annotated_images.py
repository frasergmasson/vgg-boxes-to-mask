import argparse
import json
import shutil
import os

def contains_n_regions(image,n):
    return len(image["regions"])>=n

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument('json_file')
    args = parser.parse_args()

    f = open(args.json_file)
    project = json.load(f)
    f.close()

    annotated_images = ([image["filename"] for image in project.values() if contains_n_regions(image,1)])
    for image in annotated_images:
        shutil.copy(os.path.join(args.in_dir,image), args.out_dir)