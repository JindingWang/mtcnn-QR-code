import argparse
import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torchvision.transforms as transforms
import core.config as config
from data import PnetImageLoader
from core.models import RNet, LossFn


def parse_args():
    parser = argparse.ArgumentParser(description='Train RNet')
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.RNET_end_epoch, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--gpu', dest='gpu', help='train with gpu',
                        default='', type=str)
    parser.add_argument('--restore', dest='restore', help='retore model',
                        default=config.RNET_restore, type=bool)
    parser.add_argument('--restore_model_path', dest='restore_model_path', help='retore model\'s path',
                        default=config.RNET_restore_model_path, type=str)
    args = parser.parse_args()
    return args


def train_pnet(model_store_path, annotation_file, end_epoch, frequent=200, gpu=None, restore=False, restore_model_path=None):
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    if gpu:
        net = RNet(is_train=True, use_cuda=True)
        if restore:
            net_dict = net.state_dict()
            weights = torch.load(restore_model_path)
            weights.pop("conv5_3.weight")
            weights.pop("conv5_3.bias")
            net_dict.update(weights)
            net.load_state_dict(net_dict)
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = RNet(is_train=True, use_cuda=False)
        if restore:
            net.load_state_dict(torch.load(restore_model_path))
    net = net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.RNET_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.RNET_step_size, gamma=config.RNET_gamma)

    train_loader = torch.utils.data.DataLoader(
        PnetImageLoader(annotation_file,
                        transform=transforms.Compose([
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                            transforms.ToTensor(),
                            transforms.Normalize(config.mean, config.std)
                        ]), mode="train"),
        batch_size=config.RNET_batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for epoch in range(1, end_epoch + 1):
        index = 0
        total_all_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        total_lm_loss = 0
        for iteration, (image, gt_label, gt_bbox, gt_landmark) in enumerate(train_loader):
            index += 1
            scheduler.step()
            if gpu:
                image = image.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()
            cls_pred, box_offset_pred, landmark_pred = net(image)
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            if torch.isnan(cls_loss):
                cls_loss = torch.Tensor([0.]).float().cuda()
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
            if torch.isnan(box_offset_loss):
                box_offset_loss = torch.Tensor([0.]).float().cuda()
            landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark, landmark_pred)
            if torch.isnan(landmark_loss):
                landmark_loss = torch.Tensor([0.]).float().cuda()
            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5 + landmark_loss * 0.5

            total_all_loss += all_loss.data.cpu().numpy()
            total_cls_loss += cls_loss.data.cpu().numpy()
            total_box_loss += box_offset_loss.data.cpu().numpy()
            total_lm_loss += landmark_loss.data.cpu().numpy()

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
        print("Epoch: %d, all loss: %f, cls loss: %f, box_loss: %f, lm_loss: %f"
              % (epoch, total_all_loss/index, total_cls_loss/index, total_box_loss/index, total_lm_loss/index))

        torch.save(net.state_dict(), os.path.join(model_store_path, "rnet_epoch_%d.pth" % (epoch)))
        torch.save(net, os.path.join(model_store_path, "rnet_epoch_model_%d.pkl" % (epoch)))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_pnet(config.RNET_model_store_path, config.RNET_TRAIN_IMGLIST_PATH, args.end_epoch, args.frequent,\
              args.gpu, args.restore, args.restore_model_path)