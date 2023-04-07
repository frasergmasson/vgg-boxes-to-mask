import argparse
import json
import concurrent.futures
import os
import cv2

def resize_image(image_name,input_path,output_path,size):
    image = cv2.imread(os.path.join(input_path,image_name))
    resized_image = cv2.resize(image,size,interpolation=cv2.INTER_LINEAR)
    out_file = os.path.join(output_path,image_name)
    cv2.imwrite(out_file,resized_image)
    size_data = {
        "original_size": image.shape[:2],
        "new_size": resized_image.shape[:2]
    }
    size_file = os.path.splitext(out_file)[0] + '.json'
    with open(size_file,'w') as f:
        f.write(json.dumps(size_data))
    
    print(f"Resized image written to {out_file}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    parser.add_argument('out_dir')
    parser.add_argument('height',type=int)
    parser.add_argument('width',type=int)
    parser.add_argument('-t','--max-threads',type=int,default=1)
    args = parser.parse_args()

    image_names = [file for file in os.listdir(args.in_dir) 
                   if os.path.isfile(os.path.join(args.in_dir,file)) and "JPG" in os.path.splitext(file)[1] ] 

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as thread_pool:
        thread_pool.map(lambda x:resize_image(x,args.in_dir,args.out_dir,(args.width,args.height)),image_names)