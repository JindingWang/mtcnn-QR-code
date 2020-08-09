import argparse
import cv2
import numpy as np
import sys
import os
sys.path.append("..")
from core.detect import MtcnnDetector, create_mtcnn_net
import time
import torchvision.transforms as transforms
from PIL import Image
from six.moves import cPickle
from core.utils import convert_to_square, IoU
import core.config as config
import core.vision as vision

save_path = '/home/wjd/datasets/QRcode/Rtrain'
pnet_model_file = '../save_model/pnet_epoch_20.pth'
annotation_file = os.path.join("../annotation", 'anno_train.txt')


def gen_rnet_data(data_dir, anno_file, pnet_model_file, gpu=None, vis=False):
    # load trained pnet model
    pnet, _, _ = create_mtcnn_net(p_model_path=pnet_model_file, gpu=gpu)
    mtcnn_detector = MtcnnDetector(pnet=pnet, min_face_size=24)

    # load original_anno_file, length = 12880
    with open(anno_file, 'r') as f:
        annotation = f.readlines()
    num = len(annotation)
    print("%d pics in total" % num)

    all_boxes = []
    for index in range(num):
        if index % 100 == 0:
            print("%d images done" % index)
        # obtain boxes and aligned boxes
        anno = annotation[index].split(',')
        image = cv2.imread(os.path.join(config.image_prefix, anno[0]))
        boxes, boxes_align = mtcnn_detector.detect_pnet(im=image)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        all_boxes.append(boxes_align)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    save_file = os.path.join(data_dir, "detections_%d.pkl" % (int(time.time())))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    gen_rnet_sample_data(data_dir, anno_file, save_file)


def gen_rnet_sample_data(data_dir, anno_file, det_boxs_file):

    neg_save_dir = os.path.join(data_dir, "negative")
    pos_save_dir = os.path.join(data_dir, "positive")
    part_save_dir = os.path.join(data_dir, "part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # load ground truth from annotation file
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 24

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for i in range(num_of_images):
        annotation = annotations[i].strip().split(',')
        im_idx = os.path.join(config.image_prefix, annotation[0])
        boxes = list(map(float, annotation[9:]))
        boxes = np.array([boxes], dtype=np.float32)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    f1 = open(os.path.join("../annotation", 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join("../annotation", 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join("../annotation", 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'rb')
    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3
            if np.max(Iou) < 0.3 and neg_num < 30:
                # save the examples
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # print(save_file)
                f2.write(save_file + ' 0' * 13 + '\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.6:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f' % (
                        offset_x1, offset_y1, offset_x2, offset_y2) + ' 0'*8 + '\n')
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f' % (
                        offset_x1, offset_y1, offset_x2, offset_y2) + ' 0'*8 + '\n')
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()


def gen_aug_data(data_dir, anno_file):
    size = 24
    landmark_save_dir = os.path.join(data_dir, "landmark")
    if not os.path.exists(landmark_save_dir):
        os.makedirs(landmark_save_dir)

    landmark_anno_filename = "landmark_24.txt"
    save_landmark_anno = os.path.join("../annotation", landmark_anno_filename)
    f = open(save_landmark_anno, 'w')

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images" % num)

    idx = 0
    l_idx = 0
    for annotation in annotations:
        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done" % idx)
        annotation = annotation.strip().split(',')
        im_path = os.path.join(config.image_prefix, annotation[0])

        gt_box = list(map(float, annotation[9:]))
        if len(gt_box) == 0:
            continue
        gt_box = np.array(gt_box, dtype=np.int32)
        landmark = list(map(float, annotation[1:9]))
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)
        height, width, channel = img.shape

        x1, y1, x2, y2 = gt_box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        gt_box = np.array([gt_box], dtype=np.float)
        # random shift
        count = 0
        break_time = 0
        while count < 4:
            break_time += 1
            if break_time > 100:
                break
            bbox_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(int(-w * 0.2), int(w * 0.2))
            delta_y = np.random.randint(int(-h * 0.2), int(h * 0.2))
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_1_x = (landmark[0] - nx1) / float(bbox_size)
            offset_1_y = (landmark[1] - ny1) / float(bbox_size)

            offset_2_x = (landmark[2] - nx1) / float(bbox_size)
            offset_2_y = (landmark[3] - ny1) / float(bbox_size)

            offset_3_x = (landmark[4] - nx1) / float(bbox_size)
            offset_3_y = (landmark[5] - ny1) / float(bbox_size)

            offset_4_x = (landmark[6] - nx1) / float(bbox_size)
            offset_4_y = (landmark[7] - ny1) / float(bbox_size)

            # cal iou
            iou = IoU(crop_box.astype(np.float), gt_box)
            if iou > 0.65:
                save_file = os.path.join(landmark_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' %
                        (offset_x1, offset_y1, offset_x2, offset_y2,
                         offset_1_x, offset_1_y, offset_2_x, offset_2_y, offset_3_x, offset_3_y, offset_4_x, offset_4_y))
                l_idx += 1
                count += 1
    f.close()

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
    # gen_rnet_data(save_path, annotation_file, pnet_model_file, args.gpu)
    # gen_rnet_sample_data(save_path, annotation_file, os.path.join(save_path, "detections_1595157516.pkl"))
    gen_aug_data(save_path, annotation_file)

    f1 = open(os.path.join("../annotation", 'pos_%d.txt' % 24), 'r')
    f2 = open(os.path.join("../annotation", 'neg_%d.txt' % 24), 'r')
    f3 = open(os.path.join("../annotation", 'part_%d.txt' % 24), 'r')
    f4 = open(os.path.join("../annotation", "landmark_24.txt"), 'r')
    pos = f1.readlines()
    neg = f2.readlines()
    part = f3.readlines()
    aug = f4.readlines()

    if os.path.exists(os.path.join("../annotation", "Rnet_anno_24.txt")):
        os.remove(os.path.join("../annotation", "Rnet_anno_24.txt"))

    f5 = open(os.path.join("../annotation", "Rnet_anno_24.txt"), 'w')

    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]
    base_num = min(nums)
    # base_num = 250000
    print(len(neg), len(pos), len(part), len(aug), base_num)

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