import pandas as pd
import os
from shutil import copyfile
from tqdm.notebook import tqdm
import ast
import argparse


def add_new_path(row):
    if row.is_train:
        return f"yolo_dataset/images/train/{row.img_name}"
    else:
        return f"yolo_dataset/images/valid/{row.img_name}"


def copy_file(row):
  copyfile(row.path, row.new_path)


def get_yolo_format_bbox(img_w, img_h, box):
    xc = (box['xmin'] + box['xmax']) / (2.0 * img_w)
    yc = (box['ymin'] + box['ymax']) / (2.0 * img_h)
    w = (box['xmax'] - box['xmin']) / img_w
    h = (box['ymax'] - box['ymin']) / img_h

    return [xc, yc, w, h]        


tqdm.pandas()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('-cfg', '--config', help='configuration file in .yml', required=True)
    args = parser.parse_args()

    try:
        # Remove existing directory
        if os.path.exists("yolo_dataset"):
          os.system(f"rm -r yolo_dataset")
        # Create directory structure
        os.makedirs(f"yolo_dataset/images/train")
        os.makedirs(f"yolo_dataset/images/valid")
        os.makedirs(f"yolo_dataset/labels/train")
        os.makedirs(f"yolo_dataset/labels/valid")
        print(f"Directory structure for YOLO created")

    except Exception as e:
        raise(f"An error occurred: {str(e)}")

    dataset=pd.read_csv(args.config)
    dataset['new_path'] = dataset.apply(lambda row: add_new_path(row), axis=1)

    _ = dataset.progress_apply(lambda row: copy_file(row), axis=1)
    print("New image path for train/valid created")

    for index, row in tqdm(dataset.iterrows()):
        annotations = ast.literal_eval(row.boxes)
        bboxes = []
        for bbox in annotations:
          bbox = get_yolo_format_bbox(row.width, row.height, bbox)
          bboxes.append(bbox)

        if row.is_train:
          file_name = f"yolo_dataset/labels/train/{row.id}.txt"
          os.makedirs(os.path.dirname(file_name), exist_ok=True)
        else:
          file_name = f"yolo_dataset/labels/valid/{row.id}.txt"
          os.makedirs(os.path.dirname(file_name), exist_ok=True)

        label_row=ast.literal_eval(row.numeric_labels)
        with open(file_name, 'w') as f:
          for i, bbox in enumerate(bboxes):
            label = label_row[i]
            bbox = [label]+bbox
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')

    print("Annotations in Yolo format for all images created.")







