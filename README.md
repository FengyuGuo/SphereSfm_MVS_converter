# SphereSfm_MVS_converter

This repo is a tool to convert the image and pose from SphereSfm to perspective image and coresponding pose so that user can run dense reconstruction of COLMAP.

## Dependencies

This tool requires [OpenCV](https://docs.opencv.org/4.5.5/d2/de6/tutorial_py_setup_in_ubuntu.html), [OmniCV](https://github.com/kaustubh-sadekar/OmniCV-Lib), [numpy-quaternion](https://quaternion.readthedocs.io/en/latest/).

## Run SphereSfm on dataset

Follow the instruction of [SphereSfm](https://github.com/json87/SphereSfM) to get the pose of sphere image. And export the result as txt. The folder should be like below.

>.
>├── cameras.txt 
>├── db2.db 
>├── images 
>├── images.txt 
>├── mask.png 
>├── points3D.txt 
>└── project.ini 

## Modify the config

Modify [config.json](https://github.com/FengyuGuo/SphereSfm_MVS_converter/blob/main/config.json). Include the path of the output of SphereSfm, size of equirect image and size of output perspective image.

## Convert images, poses and points to perspective format

run

> python3 convert_pts.py config.json

> python3 convert_img.py config.json

This will create persp folder in working folder of SphereSfm.

The conversion result can be visualized in COLMAP. The sphere camera looks like cubes in the reconstructed scene.

![pose_viz](https://github.com/FengyuGuo/SphereSfm_MVS_converter/blob/main/asset/cam_pose.png)

## Run dense reconstruction of COLMAP

Start COLMAP. use Import Model to import the converted data. Then run the dense reconstruction pipeline to get the dense point cloud.

Theratically the output format confines all version of COLMAP. So you can use newest version of COLMAP the run dense reconstruction.

Below is the result after stereo and fusion.

![result](https://github.com/FengyuGuo/SphereSfm_MVS_converter/blob/main/asset/result.png)