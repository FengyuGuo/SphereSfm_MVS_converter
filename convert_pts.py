import cv2
import time
import sys
from omnicv import fisheyeImgConv
import os
import numpy as np
import json
import quaternion
from omni_mod import eqruirect2persp_map, cubemap2equirect_map

config_path = sys.argv[1]

config = json.load(open(config_path, 'r'))

input_path = os.path.join(config['input_path'], 'images')
out_path = os.path.join(config['input_path'], 'persp')
mask_path = os.path.join(config['input_path'], 'mask.png')

persp_size = config['perspective_image_size']
equirect_width = config['equirect_width']
equirect_height = config['equirect_height']

if not os.path.exists(out_path):
    os.mkdir(out_path)

out_img_params = {
    'front':{'theta':0, 'phi': 0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'right':{'theta':90, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'left':{'theta':-90, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'top':{'theta':0, 'phi':90, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'down':{'theta':0, 'phi':-90, 'fov': 90, 'width': persp_size, 'height': persp_size},
    'back':{'theta':180, 'phi':0, 'fov': 90, 'width': persp_size, 'height': persp_size}
}

def rot_phi(phi):
    ret = np.eye(3)
    phi_rad = phi / 180.0 * np.pi
    ret[1, 1] = np.cos(phi_rad)
    ret[1, 2] = -np.sin(phi_rad)
    ret[2, 1] = np.sin(phi_rad)
    ret[2, 2] = np.cos(phi_rad)
    return ret

def rot_theta(theta):
    ret = np.eye(3)
    theta_rad = theta / 180.0 * np.pi
    ret[0, 0] = np.cos(theta_rad)
    ret[0, 2] = np.sin(theta_rad)
    ret[2, 0] = -np.sin(theta_rad)
    ret[2, 2] = np.cos(theta_rad)
    return ret


mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask.astype('float') / 255.0).astype('uint8')
mask = np.stack([mask, mask,  mask], 2)

cam_file = open(os.path.join(out_path, 'cameras.txt'), 'w')

map_cam_id_name = {
    0:'top', 1:'left', 2:'front', 3:'right', 4:'back', 5:'down'
}
map_cam_name_id = {
    'top':0, 'left':1, 'front':2, 'right':3, 'back':4, 'down':5
}

for name, param in out_img_params.items():
    mapx, mapy = eqruirect2persp_map((equirect_height, equirect_width), param['fov'], param['theta'], param['phi'], param['height'], param['width'])
    # print('cam name: {}'.format(name))
    R_phi = rot_phi(param['phi'])
    R_theta = rot_theta(param['theta'])
    R_cam_to_front = np.matmul(R_theta, R_phi)

    # x_cam = R_cam_to_front[:, 0]
    # y_cam = R_cam_to_front[:, 1]
    # z_cam = R_cam_to_front[:, 2]
    # print('x, y, z: {}, {}, {}'.format(x_cam, y_cam, z_cam))
    
    param['mapx'] = mapx
    param['mapy'] = mapy
    param['R_cam_to_front'] = R_cam_to_front

    cam_file.write('{} SIMPLE_PINHOLE {} {} {} {} {}\n'.format(map_cam_name_id[name], persp_size, persp_size, persp_size / 2, persp_size / 2, persp_size / 2))

# print(out_img_params)
cam_file.close()
# exit()

mapper = fisheyeImgConv()

cubemap_idx = np.zeros((persp_size * 3, persp_size * 4), dtype='uint8')

cubemap_idx[:persp_size, persp_size:persp_size*2] = 0 # top
cubemap_idx[persp_size:persp_size*2, :persp_size] = 1 # left
cubemap_idx[persp_size:persp_size*2, persp_size:persp_size * 2] = 2 # front
cubemap_idx[persp_size:persp_size*2, persp_size*2:persp_size*3] = 3 # right
cubemap_idx[persp_size:persp_size*2, persp_size*3:] = 4 # back
cubemap_idx[persp_size*2:, persp_size:persp_size*2] = 5 # down


equirect_cam = mapper.cubemap2equirect(cubemap_idx, (equirect_height, equirect_width))
equirect_viz = equirect_cam * 50

# cv2.imshow('idx', equirect_viz)
# cv2.waitKey(10000000)
# cv2.destroyAllWindows()

# dice_mapx, dice_mapy = equirect2cubemap_map(cubemap_idx.shape, 480, True)
cube_mapx, cube_mapy = cubemap2equirect_map(persp_size, [equirect_height, equirect_width])

# read in the output of colmap

img_path = os.path.join(config['input_path'], 'images.txt')
pts_path = os.path.join(config['input_path'], 'points3D.txt')

images = {}

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

img_file = open(img_path, 'r')

line = img_file.readline()
line_status = 1 # 1 for pose line, 2 for pts line

map_pt3d_id_to_2d = {}

while len(line) > 0:
    if line[0] == '#':
        line = img_file.readline()
        continue
    if line_status == 1:
        line_split = line.strip().split(' ')
        line_status = 2
        img_id = int(line_split[0])
        cur_img_id = img_id
        w = float(line_split[1])
        x = float(line_split[2])
        y = float(line_split[3])
        z = float(line_split[4])
        R_GtoC = quaternion.as_rotation_matrix(quaternion.quaternion(w, x, y, z))
        p_GinC = np.array([float(line_split[5]), float(line_split[6]), float(line_split[7])])
        img_info={}
        img_info['R_GtoC'] = R_GtoC
        img_info['p_GinC'] = p_GinC
        cam_id = int(line_split[8])
        name = line_split[9]
        img_info['cam_id'] = cam_id
        img_info['name'] = name
        images[img_id] = img_info
    elif line_status == 2:
        line_split = line.strip().split(' ')
        line_status = 1
        if len(line_split) % 3 != 0:
            print('pts line corrupted!!')
        pts = np.array(line_split, dtype='float').reshape(-1, 3)
        # pts_info = {int(row[2]): [row[0], row[1]] for row in pts if row[2] != -1}
        pts_info = []
        pts_in_new_cam = {}
        new_pts = {}
        for cam in out_img_params:
            pts_in_new_cam[cam] = 0
            new_pts[cam] = []
        for row in pts:
            pt3d_id = int(row[2])
            if pt3d_id == -1:
                continue
            xy = [row[0], row[1]]
            remaped_cam = map_cam_id_name[equirect_cam[int(xy[1]), int(xy[0])]]
            u = cv2.getRectSubPix(cube_mapx, (1, 1), xy)[0, 0]
            v = cv2.getRectSubPix(cube_mapy, (1, 1), xy)[0, 0]
            if u<0 or v<0 or u>=persp_size or v>=persp_size:
                print('discard {}'.format(pt3d_id))
                continue
            if not pt3d_id in map_pt3d_id_to_2d:
                map_pt3d_id_to_2d[pt3d_id] = {}
            pts_info.append({
                'pts3d_id':pt3d_id,
                'xy_equi':xy,
                'new_cam_name':remaped_cam,
                'xy_in_new_cam':[u, v],
                'id_in_new_cam':pts_in_new_cam[remaped_cam]
            })
            map_pt3d_id_to_2d[pt3d_id][cur_img_id] = len(pts_info) - 1 # new id that discard the invalid index -1
            pts_in_new_cam[remaped_cam] += 1
            new_pts[remaped_cam].append(len(pts_info) - 1)
        # conver the points from equirect image to cube image
        # print(pts_info)
        images[cur_img_id]['pts'] = pts_info
        images[cur_img_id]['new_pts'] = new_pts
        # exit()
    line = img_file.readline()
img_file.close()
# print(images)

# exit()

# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 30571, mean track length: 9.3540283274999183

pts_file = open(pts_path, 'r')

line = pts_file.readline()
pts3d = {}
while len(line) > 0:
    if line[0] == '#':
        line = pts_file.readline()
        continue
    line_split = line.split(' ')
    pts_id = int(line_split[0])
    xyz=[float(line_split[i]) for i in range(1, 4)]
    rgb=[int(line_split[i]) for i in range(4, 7)]
    error = float(line_split[7])
    track_info = []
    for i in range(8, len(line_split), 2):
        track_info.append([int(line_split[i]), int(line_split[i + 1])])
    pts3d[pts_id] = {}
    pts3d[pts_id]['xyz'] = xyz
    pts3d[pts_id]['rgb'] = rgb
    pts3d[pts_id]['error'] = error
    line = pts_file.readline()
pts_file.close()
# print(pts3d)

# reorganize the image and points
new_images = {}
new_img_file = open(os.path.join(out_path, 'images.txt'), 'w')
for img_id, img_info in images.items():
    print('old image id: {}'.format(img_id))
    # assign image id for new cameras
    new_cam_cnt = len(out_img_params)
    for cam_name, cam_param in out_img_params.items():
        new_img_id = img_id * new_cam_cnt + map_cam_name_id[cam_name]
        new_img_name = img_info['name'].split('.')[0] + '_{}.jpg'.format(cam_name)
        R_GtoC_new = np.matmul(cam_param['R_cam_to_front'].T, img_info['R_GtoC'])
        q_GtoC_new = quaternion.from_rotation_matrix(R_GtoC_new)
        # print(q_GtoC_new.w)
        p_CinG = -np.matmul(img_info['R_GtoC'].T, img_info['p_GinC'].reshape(3, 1))
        p_CinG_new = p_CinG
        p_GinC_new = -np.matmul(R_GtoC_new, p_CinG_new.reshape(3, 1)).squeeze()
        line = '{} {} {} {} {} {} {} {} {} {}\n'.format(new_img_id, q_GtoC_new.w, q_GtoC_new.x, q_GtoC_new.y, q_GtoC_new.z,
            p_GinC_new[0], p_GinC_new[1], p_GinC_new[2], map_cam_name_id[cam_name], new_img_name)
        new_img_file.write(line)
        pt2d_in_cam = img_info['new_pts'][cam_name] # pt2d index in current old image
        print('{} have {} points'.format(cam_name, len(pt2d_in_cam)))
        for id_2d in pt2d_in_cam:
            uv = img_info['pts'][id_2d]['xy_in_new_cam']
            pt3d_id = img_info['pts'][id_2d]['pts3d_id']
            new_img_file.write('{} {} {} '.format(uv[0], uv[1], pt3d_id))
        new_img_file.write('\n')

new_img_file.close()

new_pts3d_file = open(os.path.join(out_path, 'points3D.txt'), 'w')

for pt3d_id, pt3d_info in pts3d.items():
    pt_in_img = map_pt3d_id_to_2d[pt3d_id]
    line = '{} {} {} {} {} {} {} {} '.format(pt3d_id, pt3d_info['xyz'][0], pt3d_info['xyz'][1], pt3d_info['xyz'][2],
        pt3d_info['rgb'][0], pt3d_info['rgb'][1], pt3d_info['rgb'][2], pt3d_info['error'])
    new_pts3d_file.write(line)

    for old_img_id, pt2d_idx in pt_in_img.items():
        # get the new img id
        cam_name = images[old_img_id]['pts'][pt2d_idx]['new_cam_name']
        new_img_id = len(out_img_params) * old_img_id + map_cam_name_id[cam_name]
        new_pt2d_idx = images[old_img_id]['pts'][pt2d_idx]['id_in_new_cam']
        new_pts3d_file.write('{} {} '.format(new_img_id, new_pt2d_idx))
    new_pts3d_file.write('\n')

new_pts3d_file.close()