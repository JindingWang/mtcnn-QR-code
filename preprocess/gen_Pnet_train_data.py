"""
    2018-10-20 15:50:20
    generate positive, negative, positive images whose size are 12*12 and feed into PNet
"""
import sys
import os
import random
sys.path.append("..")
import cv2
import numpy as np
from utils import IoU
from core import config as config
from BBox_utils import BBox, rotate

def gen_imglist_pnet():
    f1 = open(os.path.join('../annotation', 'Ppos.txt'), 'r')
    f2 = open(os.path.join('../annotation', 'Pneg.txt'), 'r')
    f3 = open(os.path.join('../annotation', 'Ppart.txt'), 'r')
    f4 = open(os.path.join('../annotation', 'Paug.txt'), 'r')

    pos = f1.readlines()
    neg = f2.readlines()
    part = f3.readlines()
    aug = f4.readlines()

    output_file = os.path.join('../annotation', 'Pnet_anno_12.txt')
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f5:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        base_num = min(nums)
        # base_num = 250000
        print(len(neg), len(pos), len(part), base_num)

        # shuffle the order of the initial data
        # if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=True)
        pos_keep = np.random.choice(len(pos), size=base_num, replace=True)
        part_keep = np.random.choice(len(part), size=base_num, replace=True)
        print(len(neg_keep), len(pos_keep), len(part_keep))

        # write the data according to the shuffled order
        for i in pos_keep:
            f5.write(pos[i])
        for i in neg_keep:
            f5.write(neg[i])
        for i in part_keep:
            f5.write(part[i])
        for item in aug:
            f5.write(item)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

if __name__ == "__main__":
    anno_train_file = os.path.join("../annotation", 'anno_train.txt')
    pos_save_dir = "/home/wjd/datasets/QRcode/Ptrain/positive"
    part_save_dir = "/home/wjd/datasets/QRcode/Ptrain/part"
    neg_save_dir = "/home/wjd/datasets/QRcode/Ptrain/negative"
    aug_save_dir = "/home/wjd/datasets/QRcode/Ptrain/augment"

    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(aug_save_dir):
        os.mkdir(aug_save_dir)

    # store labels of positive, negative, part images
    f1 = open(os.path.join('../annotation', 'Ppos.txt'), 'w')
    f2 = open(os.path.join('../annotation', 'Pneg.txt'), 'w')
    f3 = open(os.path.join('../annotation', 'Ppart.txt'), 'w')
    f4 = open(os.path.join('../annotation', 'Paug.txt'), 'w')

    # anno_file: store labels of the wider face training data
    with open(anno_train_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)

    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # don't care
    aug_idx = 0
    idx = 0
    for annotation in annotations:
        annotation = annotation.split(',') # [img_name, bbox]
        im_path = os.path.join(config.image_prefix, annotation[0])
        bbox = list(map(float, annotation[9:]))
        boxes = np.array([bbox], dtype=np.float)
        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print(str(idx) + ' / ' + str(num) + "images done")

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 25:
            size = np.random.randint(12, max(min(width, height) // 2, 13)) # edge length of the square
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = IoU(crop_box, boxes)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx) # add background negative sample
                f2.write(save_file + ' 0'*13 + '\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # generate negative examples that have overlap with gt
            count = 0
            while count < 5:
                size = np.random.randint(12, max(13, min(width, height) // 2))
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                if 0.4 > np.max(Iou) > 0:
                    # Iou with all gts must below 0.3
                    cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    #add negative samples having overlap with gt
                    f2.write(save_file + ' 0'*13 + '\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                    count += 1

            # generate positive examples and part faces
            count = 0
            while count < 20:
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h))) # edge length
                # delta here is the offset of box center
                delta_x = np.random.randint(int(-w * 0.2), int(w * 0.2))
                delta_y = np.random.randint(int(-h * 0.2), int(h * 0.2))

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.7:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f' %
                             (offset_x1, offset_y1, offset_x2, offset_y2) + ' 0'*8 + '\n')
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    count += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f' %
                             (offset_x1, offset_y1, offset_x2, offset_y2) + ' 0'*8 + '\n')
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                    count += 1

        F_imgs = []
        F_landmarks = []
        gt_landmarks = list(map(float, annotation[1:9]))
        gt_landmarks = np.array(gt_landmarks).reshape(-1, 2)
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        # normalize land mark by dividing the width and height of the ground truth bounding box
        landmarks = np.zeros((4, 2))
        landmarks[:, 0] = (gt_landmarks[:, 0] - bbox[0]) / w
        landmarks[:, 1] = (gt_landmarks[:, 1] - bbox[1]) / h
        # get sub-image from bbox (x_left, y_top, x_right, y_bottom)
        f_face = img[int(bbox[1]):int(bbox[3]+1), int(bbox[0]):int(bbox[2]+1)]
        # resize the gt image to specified size
        f_face = cv2.resize(f_face, (12, 12))

        F_imgs.append(f_face)
        F_landmarks.append(landmarks.reshape(8))

        if True:
            # bbox (x_left, y_top, x_right, y_bottom)
            # random shift
            for i in range(10):
                bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                delta_x = np.random.randint(int(-w * 0.2), int(w * 0.2))
                delta_y = np.random.randint(int(-h * 0.2), int(h * 0.2))
                nx1 = int(max(bbox[0] + w / 2 - bbox_size / 2 + delta_x, 0))
                ny1 = int(max(bbox[1] + h / 2 - bbox_size / 2 + delta_y, 0))
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                crop_box = np.array([nx1, ny1, nx2, ny2])

                # cal iou
                iou = IoU(crop_box, boxes)
                if iou > 0.7:
                    cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                    resized_im = cv2.resize(cropped_im, (12, 12))
                    F_imgs.append(resized_im)
                    # normalize
                    aug_landmark = np.zeros((4, 2))
                    aug_landmark[:, 0] = (gt_landmarks[:, 0] - nx1) / bbox_size
                    aug_landmark[:, 1] = (gt_landmarks[:, 1] - ny1) / bbox_size
                    F_landmarks.append(aug_landmark.reshape(8))

                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    bbox_tmp = BBox([nx1, ny1, nx2, ny2])
                    # rotate
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox_tmp, \
                                                                         bbox_tmp.reprojectLandmark(landmark_), 5)  # 逆时针旋转
                        # landmark_offset
                        landmark_rotated = bbox_tmp.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (12, 12))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(8))

                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox_tmp, \
                                                                         bbox_tmp.reprojectLandmark(landmark_), -5)  # 顺时针旋转
                        landmark_rotated = bbox_tmp.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (12, 12))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(8))

            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
            for i in range(len(F_imgs)):
                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue
                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                save_file = os.path.join(aug_save_dir, "%d.jpg" % (aug_idx))
                cv2.imwrite(save_file, F_imgs[i])
                landmarks = map(str, list(F_landmarks[i]))
                f4.write(save_file + " -2 " + '0 '*4 + " ".join(landmarks) + "\n")
                aug_idx += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    print("done with generatint pnet datasets \n assembling four datasets......")
    gen_imglist_pnet()