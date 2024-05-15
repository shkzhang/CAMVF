import os
import re
import shutil

import cv2
import numpy as np
def cv_imread(path,flags):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return cv_img
def crop(original_image_path,mask_image_path):
    # 读取原图和掩膜图
    original_image = cv_imread(original_image_path, cv2.IMREAD_COLOR)
    mask_image = cv_imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # 获取原图和掩膜图的尺寸
    original_height, original_width, _ = original_image.shape
    mask_height, mask_width = mask_image.shape

    # 计算居中裁剪区域的坐标
    x_offset = max(0,(mask_width-original_width) // 2)
    y_offset =max(0, (mask_height-original_height ) // 2)
    print(x_offset,y_offset)
    # 创建与原图大小相同的掩膜图
    new_mask_image = np.zeros_like(original_image[:,:,0])
    new_mask_image = mask_image[y_offset:y_offset+original_height, x_offset:x_offset+original_width]

    # 根据掩膜图裁剪出区域（额外向周围多裁剪50个像素）
    extend = 50
    x, y, w, h = cv2.boundingRect(new_mask_image)
    x = max(0, x - extend)
    y = max(0, y - extend)
    w = min(original_width, x + w + extend*2) - x
    h = min(original_height, y + h + extend*2) - y
    new_mask_image = np.zeros_like(original_image[:,:,0])
    new_mask_image[y:y+h, x:x+w] = 255
    return new_mask_image


root_dir = r"E:\data\参量成像\HCC"
for case_name in os.listdir(root_dir):
    case_path = os.path.join(root_dir,case_name)
    mask_path = os.path.join(case_path,"Mask2.png")
    pca_path = os.path.join(case_path,"pca")
    img_name_list=sorted(os.listdir(pca_path),key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)])
    image_up = os.path.join(pca_path,img_name_list[0])
    image_down = os.path.join(pca_path,img_name_list[1])
    mask = crop(image_up,mask_path)
    cv2.imencode('.png', mask)[1].tofile(os.path.join(pca_path,'Mask.png'))
    shutil.copy(image_up,os.path.join(pca_path,'Image_UP.png'))
    shutil.copy(image_down,os.path.join(pca_path,'Image_DOWN.png'))
