import cv2
import json
import glob
from pprint import pprint
import os

if __name__ == '__main__':
    root = "data/football"
    output_path = "football_yolo"
    video_paths = list(glob.iglob("{}/*/*.mp4".format(root)))
    anno_paths = list(glob.iglob("{}/*/*.json".format(root)))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "train", "images"))
        os.mkdir(os.path.join(output_path, "train", "labels"))

    for video_id, (video_path, anno_path) in enumerate(zip(video_paths, anno_paths)):
        counter = 1
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        with open(anno_path, "r") as json_file:
            json_data = json.load(json_file)
        image_width = json_data["images"][0]["width"]
        image_height = json_data["images"][0]["height"]
        while video.isOpened():
            flag, frame = video.read()
            if not flag:
                break
            cv2.imwrite(os.path.join(output_path, "images", "{}_{}.jpg".format(video_id+1, counter)), frame)
            objects = [item for item in json_data["annotations"] if item["image_id"] == counter and item["category_id"] > 2]
            with open(os.path.join(output_path, "labels", "{}_{}.txt".format(video_id+1, counter)), "w") as txt_file:
                for obj in objects:
                    bbox = obj["bbox"]
                    xmin, ymin, width, height = bbox
                    xmin /= image_width
                    ymin /= image_height
                    width /= image_width
                    height /= image_height
                    x_cent = xmin + width/2
                    y_cent = ymin + height/2
                    if obj["category_id"] == 3:
                        cls = 0
                    else:
                        cls = 1
                    txt_file.write("{} {:6f} {:6f} {:6f} {:6f}\n".format(cls, x_cent, y_cent, width, height))


                    # TEST ONLY
            #         xmax = xmin + width
            #         ymax = ymin + height
            #         xmin *= image_width
            #         ymin *= image_height
            #         xmax*= image_width
            #         ymax *= image_height
            #         cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 255), thickness=1)
            # cv2.imwrite(os.path.join(output_path, "images", "{}_{}.jpg".format(video_id+1, counter)), frame)
            counter += 1







