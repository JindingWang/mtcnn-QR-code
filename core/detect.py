import cv2
import time
import numpy as np
import torch
from torch.autograd.variable import Variable
from core.models import PNet, RNet, ONet
import core.utils as utils
import core.image_tools as image_tools


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, gpu=None):
    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        if gpu:
            pnet = PNet(is_train=False, use_cuda=True)
            pnet = torch.nn.DataParallel(pnet).cuda()
            pnet.load_state_dict(torch.load(p_model_path))
        else:
            # forcing all GPU tensors to be in CPU while loading
            pnet = PNet(is_train=False, use_cuda=False)
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

    if r_model_path is not None:
        if gpu:
            rnet = RNet(is_train=False, use_cuda=True)
            rnet = torch.nn.DataParallel(rnet).cuda()
            rnet.load_state_dict(torch.load(r_model_path))
        else:
            rnet = RNet(is_train=False, use_cuda=False)
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        if gpu:
            onet = ONet(is_train=False, use_cuda=True)
            onet = torch.nn.DataParallel(onet).cuda()
            onet.load_state_dict(torch.load(o_model_path))
        else:
            onet = ONet(is_train=False, use_cuda=False)
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

    return pnet, rnet, onet


class MtcnnDetector(object):
    def  __init__(self, pnet = None, rnet = None, onet = None,
                 min_face_size=24, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.709,):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def unique_image_format(self, im):
        if not isinstance(im,np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    def square_bbox(self, bbox, width, height):
        square_bbox = bbox.copy()
        num_box = bbox.shape[0]
        for i in range(num_box):
            bbox[i, 0], bbox[i, 2] = min(bbox[i, 0], bbox[i, 2]), max(bbox[i, 0], bbox[i, 2])
            bbox[i, 1], bbox[i, 3] = min(bbox[i, 1], bbox[i, 3]), max(bbox[i, 1], bbox[i, 3])

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        for i in range(num_box):
            square_bbox[i, 0] = min(max(0, square_bbox[i, 0]), width-5)
            square_bbox[i, 1] = min(max(0, square_bbox[i, 1]), height-5)
            square_bbox[i, 2] = max(0, min(width-1, square_bbox[i, 2]))
            square_bbox[i, 3] = max(0, min(height-1, square_bbox[i, 3]))
        return square_bbox

    def generate_bounding_box(self, map, reg, scale, threshold):
        """
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
        """
        stride = 2
        cellsize = 12 # receptive field

        t_index = np.where(map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        # reg = (1, n, m, 4)
        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1], 0]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),       # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),       # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale), # x2
                                 np.round((stride * t_index[0] + cellsize) / scale), # y2
                                 score,
                                 reg])

        return boundingbox.T

    def resize_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized

    def detect_pnet(self, im):
        net_size = 12
        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = self.resize_image(im, current_scale) # scale = 1.0
        im_resized = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        while min(current_height, current_width) > net_size:
            feed_imgs = []
            image_tensor = image_tools.convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)
            feed_imgs = Variable(feed_imgs)

            feed_imgs = feed_imgs.cuda()

            # receptive field is 12×12, imaging the image is segmented into many 12*12 grid, reshaping the result can
            # generate the cls and reg result of every grid.
            # 12×12 --> score   12×12 --> bounding box
            cls_map, reg_map = self.pnet_detector(feed_imgs)

            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg_map.cpu())

            # boxes = [x1, y1, x2, y2, score, reg]
            boxes = self.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.6, 'Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack([all_boxes[:,0],
                   all_boxes[:,1],
                   all_boxes[:,2],
                   all_boxes[:,3],
                   all_boxes[:,4]])

        boxes = boxes.T

        # boxes = boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4]])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        h, w, c = im.shape

        if dets is None:
            return None, None
        # return square boxes
        dets = self.square_bbox(dets, w, h)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = im[int(dets[i, 1]):int(dets[i, 3] + 1), int(dets[i, 0]):int(dets[i, 2] + 1), :]
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im = cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB)
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))
        feed_imgs = feed_imgs.cuda()

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.6)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        boxes = np.vstack([keep_boxes[:,0],
                              keep_boxes[:,1],
                              keep_boxes[:,2],
                              keep_boxes[:,3],
                              keep_cls[:,0]])

        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_topx,
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0]])

        boxes = boxes.T
        boxes_align = boxes_align.T
        return boxes, boxes_align

    def detect_onet(self, im, dets):
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = self.square_bbox(dets, w, h)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = im[int(dets[i, 1]):int(dets[i, 3] + 1), int(dets[i, 0]):int(dets[i, 2] + 1), :]
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im = cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB)
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))
        feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = np.round(cls_map.cpu().data.numpy(), 3)
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]
        max_cls = np.max(cls_map)
        if np.sum(keep_inds) == 0:
            keep_inds = np.where(cls_map == max_cls)[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])

        boxes_align = boxes_align.T

        landmark = np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 ])

        landmark_align = landmark.T

        return boxes_align, landmark_align

    def detect_face(self,img):
        """Detect face over image
        """
        boxes_align = np.array([])
        landmark_align =np.array([])

        t = time.time()
        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])
        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])
        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

        #t3 = time.time() - t
        #print("time cost " + '{:.3f}'.format(t3))

        return boxes_align, landmark_align