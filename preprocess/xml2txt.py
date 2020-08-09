import os
import random
from xml.etree import ElementTree as ET

if __name__ == "__main__":
    f1 = open('..\\dataset\\annotation\\anno_train.txt', 'w')
    f2 = open('..\\dataset\\annotation\\anno_val.txt', 'w')
    anno_dir = "C:\\Users\\damon\\Desktop\\QRcode\\label_xml"

    code_anno = "C:\\Users\\damon\\Desktop\\QRcode\\labels.txt"
    with open(code_anno, 'r') as f:
        code = [x.strip().split(',')[:-1] for x in f.readlines()]

    num = len(code)
    for i in range(num):
        xml_name = code[i][0] + '.xml'
        anno = ET.parse(os.path.join(anno_dir, xml_name))
        objs = anno.findall('object')
        num_objs = len(objs)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            code[i] = code[i] + [str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2))]

    random.shuffle(code)
    for idx, list_mem in enumerate(code):
        if idx < num * 0.9:
            f1.write(','.join(list_mem) + "\n")
        else:
            f2.write(','.join(list_mem) + "\n")

    f1.close()
    f2.close()


"""
    anno_list = os.listdir(anno_dir)
    anno_list.sort()
    num = len(anno_list)
    rst = []

    for i in range(num):
        img_name = anno_list[i][:-4]
        anno = ET.parse(os.path.join(anno_dir, anno_list[i]))
        objs = anno.findall('object')
        num_objs = len(objs)
        rst.append([img_name])
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            rst[i] = rst[i] + [str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2))]

    random.shuffle(rst)
    for idx, list_mem in enumerate(rst):
        if idx < num * 0.9:
            f1.write(','.join(list_mem) + "\n")
        else:
            f2.write(','.join(list_mem) + "\n")

    f1.close()
    f2.close()
"""