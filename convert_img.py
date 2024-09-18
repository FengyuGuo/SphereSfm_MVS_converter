import cv2
import time
import sys
from omnicv import fisheyeImgConv
import os
import numpy as np
import json
from omni_mod import eqruirect2persp_map

config_path = sys.argv[1]

config = json.load(open(config_path, 'r'))

input_path = os.path.join(config['input_path'], 'images')
out_path = os.path.join(config['input_path'], 'persp', 'images')
mask_path = os.path.join(config['input_path'], 'mask.png')

persp_size = config['perspective_image_size']

out_img_params = {
    'front':{'theta':0, 'phi': 0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'right':{'theta':90, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'left':{'theta':-90, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'top':{'theta':0, 'phi':90, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'down':{'theta':0, 'phi':-90, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'back':{'theta':180, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size}
}

if not os.path.exists(out_path):
    os.mkdir(out_path)

imgs = None

for _, __, files in os.walk(input_path):
    imgs = files
    break

imgs.sort()

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask.astype('float') / 255.0).astype('uint8')
mask = np.stack([mask, mask,  mask], 2)
# print(mask.shape, mask.dtype)

for name, param in out_img_params.items():
    mapx, mapy = eqruirect2persp_map((mask.shape[0], mask.shape[1]), param['fov'], param['theta'], param['phi'], param['height'], param['width'])
    param['mapx'] = mapx
    param['mapy'] = mapy

for f in files:
    img_path = os.path.join(input_path, f)
    equiRect = cv2.imread(img_path)
    masked = equiRect * mask
    for name, param in out_img_params.items():
        out_name = f.split('.')[0] + '_{}.jpg'.format(name)
        img_out_path = os.path.join(out_path, out_name)
        
        persp = cv2.remap(masked, param['mapx'], param['mapy'], cv2.INTER_CUBIC)
        cv2.imwrite(img_out_path, persp)
        print('saving {} .....'.format(img_out_path))

