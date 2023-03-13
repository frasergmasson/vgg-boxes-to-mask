import argparse
import json

def contains_n_regions(image,n):
    return len(image["regions"])>=n

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    f = open(args.json_file)
    project = json.load(f)
    f.close()

    annotated_images = ([image["filename"] for image in project.values() if contains_n_regions(image,1)])
    multiple_annotations = len([image["filename"] for image in project.values() if contains_n_regions(image,2)])
    print(f"Images with one or more annotations: {annotated_images}")
    print(f"Total: {len(annotated_images)}")