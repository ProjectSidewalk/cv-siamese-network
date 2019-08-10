
#%%
import os
import pandas as pd
from utils import *
from PIL import Image
import copy
from queue import Queue
import threading
import time


#%%
path_to_gsv_scrapes = os.path.join("..","dc_scrapes_dump")

root_path = "data"

train_csv_path = "Test.csv"
train_crop_path = os.path.join(root_path, "test")
train_df = pd.read_csv(train_csv_path).sort_values(by="Pano ID")
train_df.head(10)


#%%
def bulk_extract_crops(path_to_crop_csv, destination_dir, path_to_gsv_scrapes,num_threads=4):
    '''
    takes a csv of rows:
    Pano ID, SV_x, SV_y, Label, Photog Heading, Heading, Label ID 
    and get depth-proportioned crops around each features described by each row
    writes each crop to a file in a directory within destination_dir named by that label
    '''
    crop_queue = Queue(0)
    csv_file = open(path_to_crop_csv)
    csv_f = csv.reader(csv_file)
    temp = csv.reader(open(path_to_crop_csv))
    total_crops = sum(1 for line in temp)
    counter = 0
    for row in csv_f:
        # skip header row
        if counter == 0:
            counter += 1
            continue
        crop_queue.put(row)
    
    c_threads = []
    for i in range(0, num_threads):
        c_threads.append(threading.Thread(target=crop_thread, args=(crop_queue,destination_dir,total_crops), daemon=True))
        c_threads[i].start()

    while not crop_queue.empty():
        time.sleep(1)

    for i in range(0, num_threads):
        c_threads[i].join(1)

    print("Finished.")

def crop_thread(to_crop,destination_dir,total_size):
    while not to_crop.empty():
        row = to_crop.get()
        row_num = to_crop.qsize()
        to_crop.task_done()
        pano_id = row[0]

        sv_image_x = float(row[1])
        sv_image_y = float(row[2])
        label_type = row[3]
        photographer_heading = float(row[4])

        destination_basename = '{}crop{},{}'.format(pano_id, sv_image_x, sv_image_y)
        crop_destination = os.path.join(destination_dir, str(label_type), destination_basename)
        if not os.path.exists(crop_destination+".jpg"):
            pano_root = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id)
            pano_img_path =  pano_root + ".jpg"
            path_to_depth = pano_root+".depth.txt"
            pano_yaw_deg = 180 - photographer_heading

            # Extract the crop
            if os.path.exists(path_to_depth):
                destination_folder = os.path.join(destination_dir, str(label_type))
                if not os.path.isdir(destination_folder):
                    os.makedirs(destination_folder)

                try:
                    depth = None
                    GSV_IMAGE_WIDTH,GSV_IMAGE_HEIGHT = (None,None)
                    pano_img = None

                    with open(path_to_depth, 'rb') as f:
                        depth = np.loadtxt(f)
                    GSV_IMAGE_WIDTH,GSV_IMAGE_HEIGHT = extract_width_and_height(pano_root+".xml")
                    pano_img = Image.open(pano_img_path)

                    make_single_crop(pano_img, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, depth, pano_id, sv_image_x, sv_image_y, pano_yaw_deg, crop_destination)
                    print( "Successfully extracted crop to {} {}/{}".format(destination_basename,row_num,total_size))
                except Exception as e:
                    print( "Cropping {} at {},{} failed.".format(pano_id, sv_image_x, sv_image_y) )
                    print(e)
            else:
                print( "Panorama image not found for {} at {}".format(pano_id, pano_img_path) )


#%%
bulk_extract_crops(train_csv_path, train_crop_path, path_to_gsv_scrapes, num_threads=12)
#%%
print("Done")