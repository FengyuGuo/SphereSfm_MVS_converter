import cv2
import numpy as np

def eqruirect2persp_map(
                    img_shape,
                    FOV,
                    THETA,
                    PHI,
                    Hd,
                    Wd
                    ):

    # THETA is left/right angle, PHI is up/down angle, both in degree
    equ_h, equ_w = img_shape

    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    wFOV = FOV
    hFOV = float(Hd) / Wd * wFOV

    c_x = (Wd) / 2.0
    c_y = (Hd) / 2.0

    w_len = 2 * np.tan(np.radians(wFOV / 2.0))
    w_interval = w_len / (Wd)

    h_len = 2 * np.tan(np.radians(hFOV / 2.0))
    h_interval = h_len / (Hd)

    x_map = np.zeros([Hd, Wd], np.float32) + 1
    y_map = np.tile((np.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
    z_map = -np.tile((np.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

    xyz = np.zeros([Hd, Wd, 3], np.float)
    xyz[:, :, 0] = (x_map / D)[:, :]
    xyz[:, :, 1] = (y_map / D)[:, :]
    xyz[:, :, 2] = (z_map / D)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([Hd * Wd, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2] / 1)
    lon = np.zeros([Hd * Wd], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([Hd, Wd]) / np.pi * 180
    lat = -lat.reshape([Hd, Wd]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    # persp = cv2.remap(img,
    #                     lon.astype(np.float32),
    #                     lat.astype(np.float32),
    #                     cv2.INTER_CUBIC,
    #                     borderMode=cv2.BORDER_WRAP)

    return lon.astype(np.float32), lat.astype(np.float32)


def equirect2cubemap_map(
                        img_shape,
                        side=256,
                        dice=False
                        ):

    inShape = img_shape
    mesh = np.stack(
        np.meshgrid(
            np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
            -np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
        ),
        -1,
    )

    # Creating a matrix that contains x,y,z values of all 6 faces
    facesXYZ = np.zeros((side, side * 6, 3), np.float32)

    # if modif:
    #     # Front face (z = 0.5)
    #     facesXYZ[:, 0 * side: 1 * side, [0, 2]] = mesh
    #     facesXYZ[:, 0 * side: 1 * side, 1] = -0.5

    #     # Right face (x = 0.5)
    #     facesXYZ[:, 1 * side: 2 * side, [1, 2]] = np.flip(mesh, axis=1)
    #     facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

    #     # Back face (z = -0.5)
    #     facesXYZ[:, 2 * side: 3 * side, [0, 2]] = mesh
    #     facesXYZ[:, 2 * side: 3 * side, 1] = 0.5

    #     # Left face (x = -0.5)
    #     facesXYZ[:, 3 * side: 4 * side, [1, 2]] = np.flip(mesh, axis=1)
    #     facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

    #     # Up face (y = 0.5)
    #     facesXYZ[:, 4 * side: 5 * side, [0, 1]] = mesh[::-1]
    #     facesXYZ[:, 4 * side: 5 * side, 2] = 0.5

    #     # Down face (y = -0.5)
    #     facesXYZ[:, 5 * side: 6 * side, [0, 1]] = mesh
    #     facesXYZ[:, 5 * side: 6 * side, 2] = -0.5

    # else:
    # Front face (z = 0.5)
    facesXYZ[:, 0 * side: 1 * side, [0, 1]] = mesh
    facesXYZ[:, 0 * side: 1 * side, 2] = 0.5

    # Right face (x = 0.5)
    facesXYZ[:, 1 * side: 2 * side, [2, 1]] = mesh
    facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

    # Back face (z = -0.5)
    facesXYZ[:, 2 * side: 3 * side, [0, 1]] = mesh
    facesXYZ[:, 2 * side: 3 * side, 2] = -0.5

    # Left face (x = -0.5)
    facesXYZ[:, 3 * side: 4 * side, [2, 1]] = mesh
    facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

    # Up face (y = 0.5)
    facesXYZ[:, 4 * side: 5 * side, [0, 2]] = mesh
    facesXYZ[:, 4 * side: 5 * side, 1] = 0.5

    # Down face (y = -0.5)
    facesXYZ[:, 5 * side: 6 * side, [0, 2]] = mesh
    facesXYZ[:, 5 * side: 6 * side, 1] = -0.5

    # Calculating the spherical coordinates phi and theta for given XYZ
    # coordinate of a cube face
    x, y, z = np.split(facesXYZ, 3, axis=-1)
    # phi = tan^-1(x/z)
    phi = np.arctan2(x, z)
    # theta = tan^-1(y/||(x,y)||)
    theta = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

    h, w = inShape
    # Calculating corresponding coordinate points in
    # the equirectangular image
    eqrec_x = (phi / (2 * np.pi) + 0.5) * w
    eqrec_y = (-theta / np.pi + 0.5) * h
    # Note: we have considered equirectangular image to
    # be mapped to a normalised form and then to the scale of (pi,2pi)

    map_x = eqrec_x
    map_y = eqrec_y

    # dstFrame = cv2.remap(srcFrame,
    #                         map_x,
    #                         map_y,
    #                         interpolation=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT)
    

    if dice:
        dice_map_x = np.zeros((side * 3, side * 4), dtype='float32')
        dice_map_y = np.zeros((side * 3, side * 4), dtype='float32')
        dice_map_x[:side, side:side*2] = cv2.flip(map_x[:, 4 * side : 5 * side, 0], 0)
        dice_map_y[:side, side:side*2] = map_y[:, 4 * side : 5 * side, 0]

        dice_map_x[side:side*2, :side] = map_x[:, 3 * side: 4 * side, 0]
        dice_map_y[side:side*2, :side] = map_y[:, 3 * side: 4 * side, 0]

        dice_map_x[side:side*2, side:side*2] = map_x[:, :side, 0]
        dice_map_y[side:side*2, side:side*2] = map_y[:, :side, 0]

        dice_map_x[side:side*2, side*2:side*3] = cv2.flip(map_x[:, side:2*side,0],1)
        dice_map_y[side:side*2, side*2:side*3] = map_y[:, side:2*side,0]

        dice_map_x[side:side*2, side*3:] = cv2.flip(map_x[:, 2 * side: 3 * side, 0], 1)
        dice_map_y[side:side*2, side*3:] = map_y[:, 2 * side: 3 * side, 0]

        dice_map_x[side*2:, side:side*2] = map_x[:, 5 * side: 6 * side, 0]
        dice_map_y[side*2:, side:side*2] = map_y[:, 5 * side: 6 * side, 0]
        return dice_map_x, dice_map_y
        # dstFrame = cv2.remap(srcFrame,
        #                     dice_map_x,
        #                     dice_map_y,
        #                     interpolation=cv2.INTER_LINEAR,
        #                     borderMode=cv2.BORDER_CONSTANT)
    else:
        return map_x, map_y

def cubemap2equirect_map(img_size, outShape):
    h = outShape[0]
    w = outShape[1]
    face_w = img_size

    phi = np.linspace(-np.pi, np.pi, num=outShape[1], dtype=np.float32)
    theta = np.linspace(np.pi, -np.pi, num=outShape[0], dtype=np.float32) / 2

    phi, theta = np.meshgrid(phi, theta)

    tp = np.zeros((h, w), dtype=np.int32)
    tp[:, : w // 8] = 2
    tp[:, w // 8: 3 * w // 8] = 3
    tp[:, 3 * w // 8: 5 * w // 8] = 0
    tp[:, 5 * w // 8: 7 * w // 8] = 1
    tp[:, 7 * w // 8:] = 2

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1

    mask = np.roll(mask, w // 8, 1)

    mask = np.concatenate([mask] * 4, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    tp = tp.astype(np.int32)

    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = tp == i
        coor_x[mask] = 0.5 * np.tan(phi[mask] - np.pi * i / 2)
        coor_y[mask] = (
            -0.5 * np.tan(theta[mask]) / np.cos(phi[mask] - np.pi * i / 2)
        )

    mask = tp == 4
    c = 0.5 * np.tan(np.pi / 2 - theta[mask])
    coor_x[mask] = c * np.sin(phi[mask])
    coor_y[mask] = c * np.cos(phi[mask])

    mask = tp == 5
    c = 0.5 * np.tan(np.pi / 2 - np.abs(theta[mask]))
    coor_x[mask] = c * np.sin(phi[mask])
    coor_y[mask] = -c * np.cos(phi[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    map_x = coor_x.astype(np.float32)
    map_y = coor_y.astype(np.float32)

    return map_x, map_y