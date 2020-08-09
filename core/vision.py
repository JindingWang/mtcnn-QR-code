import os
import sys
import cv2
sys.path.append(os.getcwd())


def vis_face(im_array, dets, landmarks, save_name):
    for i in range(dets.shape[0]):
        bbox = dets[i, :4]

        cv2.rectangle(im_array, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 5)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((4, 2))
            for j in range(4):
                cv2.circle(im_array, (int(landmarks_one[j, 0]), int(landmarks_one[j, 1])), 5, (255,0,0))  # 26

    return im_array