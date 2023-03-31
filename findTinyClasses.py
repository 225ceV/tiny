import os
import argparse
import numpy as np
import pandas as pd

coco_classes = [
     u'person',
     u'bicycle',
     u'car',
     u'motorcycle',
     u'airplane',
     u'bus',
     u'train',
     u'truck',
     u'boat',
     u'traffic light',
     u'fire hydrant',
     u'stop sign',
     u'parking meter',
     u'bench',
     u'bird',
     u'cat',
     u'dog',
     u'horse',
     u'sheep',
     u'cow',
     u'elephant',
     u'bear',
     u'zebra',
     u'giraffe',
     u'backpack',
     u'umbrella',
     u'handbag',
     u'tie',
     u'suitcase',
     u'frisbee',
     u'skis',
     u'snowboard',
     u'sports ball',
     u'kite',
     u'baseball bat',
     u'baseball glove',
     u'skateboard',
     u'surfboard',
     u'tennis racket',
     u'bottle',
     u'wine glass',
     u'cup',
     u'fork',
     u'knife',
     u'spoon',
     u'bowl',
     u'banana',
     u'apple',
     u'sandwich',
     u'orange',
     u'broccoli',
     u'carrot',
     u'hot dog',
     u'pizza',
     u'donut',
     u'cake',
     u'chair',
     u'couch',
     u'potted plant',
     u'bed',
     u'dining table',
     u'toilet',
     u'tv',
     u'laptop',
     u'mouse',
     u'remote',
     u'keyboard',
     u'cell phone',
     u'microwave',
     u'oven',
     u'toaster',
     u'sink',
     u'refrigerator',
     u'book',
     u'clock',
     u'vase',
     u'scissors',
     u'teddy bear',
     u'hair drier',
     u'toothbrush']

def getFileList(dir: str, extract: str) -> list:
    fileList = []
    filenames = os.listdir(dir)
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == extract:
            fileList.append(filename)
    return fileList


if __name__ == "__main__":
    nc = 80
    classcounter = np.zeros((80, 3))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", help="dataset path", type=str, default="../datasets/coco/labels/train2017/"
    )

    parser.add_argument("-s", "--size", help="image size", type=int, default=640)
    parser.add_argument("-l", "--limit", help="limit of pixels", type=int, default=8)
    args = parser.parse_args()
    current_root = os.getcwd()
    path = args.dataset
    size = args.size
    limit = args.limit
    fileList = getFileList(path, ".txt")
    limit = limit * limit
    counter = 0
    labelcounter = 0
    # change dir
    os.chdir(path)
    for file in fileList:
        with open(file) as f:
            for line in f:
                labelcounter += 1
                temp = line.split(" ")
                classcounter[int(temp[0]), 0] += 1
                w = float(temp[3]) * size
                h = float(temp[4]) * size
                pixels = round(w * h)
                if pixels <= limit:
                    classcounter[int(temp[0]), 1] += 1
                    counter += 1

    print(f"小目标数量：{counter}")
    print(f"目标总数：{labelcounter}")
    print(f"文件总数{len(fileList)}")
    # for i, cls in enumerate(coco_classes):
    #     print(f"{cls}\t\t总数：{classcounter[i, 0]}\t小目标：{classcounter[i, 1]}")
    classcounter[:,2] = classcounter[:,1]/classcounter[:,0]
    table = pd.DataFrame(data=classcounter, index=coco_classes, columns=['all', 'tiny_obj', 'tiny_rate'])
    table.sort_values(by=['tiny_rate', 'tiny_obj'], ascending=False, inplace=True)
    filepath = os.path.join(current_root, "hist", f"{limit}.csv")
    table.to_csv(filepath)
