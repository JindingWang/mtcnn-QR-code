import os
import cv2
import numpy as np
#import config as config

if __name__ == "__main__":
    img_dir = "/home/wjd/datasets/QRcode/images/"#config.image_prefix
    img_list = os.listdir(img_dir)
    num = len(img_list)
    R_channel = 0
    G_channel = 0
    B_channel = 0
    R_channel_square = 0
    G_channel_square = 0
    B_channel_square = 0
    pixels_num = 0

    for i in range(num):
        if not img_list[i].endswith(".jpg"):
            continue
        try:
            img = cv2.imread(os.path.join(img_dir, img_list[i]))
        except:
            print(img_list[i])
        try:
            img = np.asarray(img)
        except:
            print(img_list[i])
        [h, w, c] = img.shape
        pixels_num += h * w

        R_temp = img[:, :, 2]
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))
        G_temp = img[:, :, 1]
        G_channel += np.sum(G_temp)
        G_channel_square += np.sum(np.power(G_temp, 2.0))
        B_temp = img[:, :, 0]
        B_channel = B_channel + np.sum(B_temp)
        B_channel_square += np.sum(np.power(B_temp, 2.0))

    R_mean = R_channel / pixels_num
    G_mean = G_channel / pixels_num
    B_mean = B_channel / pixels_num

    R_std = np.sqrt(R_channel_square / pixels_num - R_mean * R_mean)
    G_std = np.sqrt(G_channel_square / pixels_num - G_mean * G_mean)
    B_std = np.sqrt(B_channel_square / pixels_num - B_mean * B_mean)

    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))
