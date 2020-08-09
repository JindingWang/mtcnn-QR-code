import sys
import os
sys.path.append("..")
import cv2
import numpy as np
import argparse
from core.detect import create_mtcnn_net, MtcnnDetector
from core.vision import vis_face
import core.config as config
from core.utils import convert_to_square, IoU
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet')
    parser.add_argument('--gpu', dest='gpu', help='train with gpu',
                        default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pnet, rnet, onet = create_mtcnn_net(p_model_path="../save_model/pnet_epoch_25.pth",
                                        r_model_path="../save_model/rnet_epoch_25.pth",
                                        o_model_path="../save_model/onet_epoch_25.pth", gpu=args.gpu)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    anno_file = os.path.join("../annotation", "anno_val.txt")

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images to test" % num)

    idx = 0
    l_idx = 0
    AP = 0
    t = time.time()
    for annotation in annotations:
        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done" % idx)
        annotation = annotation.strip().split(',')
        im_path = os.path.join(config.image_prefix, annotation[0])

        gt_box = list(map(float, annotation[9:]))
        if len(gt_box) == 0:
            continue
        gt_box = np.array([gt_box], dtype=np.int32)
        landmark = list(map(float, annotation[1:9]))
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)
        height, width, channel = img.shape

        bboxs, landmarks = mtcnn_detector.detect_face(img)
        count = 0
        for i in range(bboxs.shape[0]):
            bbox = bboxs[i, :4]
            iou = IoU(bbox.astype(np.float), gt_box)
            if (iou > 0.7):
                count += 1
        if count != 0:
            AP += 1

    t2 = time.time()
    print(AP/idx)
    print(t2-t)


    #img = vis_face(img, bboxs, landmarks, save_name)
    #cv2.imwrite(save_name, img)