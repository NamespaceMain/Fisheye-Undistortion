import cv2
import numpy as np

"""
Code by @shusen

This script a set of method to undistort fisheye image which has K1, K2, K3, K4 distortion param.

These methods include:

    1. Opencv fisheye undistort method by points.
    2. Custom fisheye undistort method by points.
    3. Opencv fisheye undistort method by image. 

    The resulting image are saved in the opencv Mat.

Arguments:
    --ori_image: Original fisheye image.
    --target_size: Undistort image size, order as [width, height].
    --intrins: Camera intrinsic matrix.
    --distort_params: input vector of distortion coefficients (K1, K2, K3, K4).
    --use_opencv: Whether to use Opencv lib.
"""

def undistort_process_by_point(ori_image, target_size, intrins, distort_params, use_opencv):

    ori_height, ori_width, _ = ori_image.shape
    image = cv2.resize(ori_image, target_size)
    resize_height, resize_width, _  = image.shape
    scale_w = resize_width/ori_width
    scale_h = resize_height/ori_height
    intrins[0,:] = intrins[0,:]*scale_w
    intrins[1,:] = intrins[1,:]*scale_h

    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(resize_width), np.arange(resize_height))
    ret_undis_points = list()

    if not use_opencv:
        x = (x - intrins[0,2]) / intrins[0,0]
        y = (y - intrins[1,2]) / intrins[1,1]
        theta_d = np.sqrt(x*x + y*y)
        for h_i in range(resize_height):
            for w_i in range(resize_width):
                coefs = [-theta_d[h_i, w_i],1,0,distort_params[0],0,distort_params[1],0,distort_params[2],0,distort_params[3]]
                coefs = np.poly1d(coefs[::-1])
                r = np.roots(coefs)
                r = r[np.isreal(r)]
                r = r[np.where(r>0.0)]
                r = [float(t) for t in r]
                
                theta = r[0]
                ttt = theta * (
                1 + distort_params[0] * np.power(theta, 2) + distort_params[1] * np.power(theta, 4) + 
                distort_params[2] * np.power(theta, 6) + distort_params[3] * np.power(theta, 8))
                
                try:
                    assert(len(r) == 1), "len r not equal 1."
                    theta = r[0]
                    r = np.tan(theta)
                    # if theta < 1.4:
                    undis_points_x = x[h_i, w_i] * r / theta_d[h_i, w_i]
                    undis_points_y = y[h_i, w_i] * r / theta_d[h_i, w_i]
                except:
                    undis_points_x = 0.0
                    undis_points_y = 0.0

                px = undis_points_x * intrins[0,0] + intrins[0,2]
                py = undis_points_y * intrins[1,1] + intrins[1,2]
                ret_undis_points.append([px,py])
    else:
        for h_i in range(resize_height):
            for w_i in range(resize_width):

                np_point = np.array([[[x[h_i,w_i], y[h_i,w_i]]]], dtype=np.float64)
                undis_points = cv2.fisheye.undistortPoints(np_point, intrins, distort_params)

                px = undis_points[0,0,0] * intrins[0,0] + intrins[0,2]
                py = undis_points[0,0,1] * intrins[1,1] + intrins[1,2]

                ret_undis_points.append([px,py])

    ret_undis_points = np.array(ret_undis_points)
    if not use_opencv:
        ret_undis_points += 2000
    else:
        ret_undis_points += abs(ret_undis_points.min())

    print("Min:{:.2},Max:{:.2}".format(ret_undis_points.min(),ret_undis_points.max()))
    new_img_size = 5000
    if not use_opencv:
        new_img = np.zeros((new_img_size,new_img_size,3), dtype=np.uint8)
    else:
        new_img = np.zeros((int(ret_undis_points.max())+1,int(ret_undis_points.max())+1,3), dtype=np.uint8)

    id = 0
    for h_i in range(resize_height):
        for w_i in range(resize_width):
            if(int(ret_undis_points[id][1]) < new_img_size and int(ret_undis_points[id][1]) >= 0 
               and int(ret_undis_points[id][0]) < new_img_size and int(ret_undis_points[id][0]) >= 0):
                
                new_img[int(ret_undis_points[id][1]),int(ret_undis_points[id][0])] = image[h_i,w_i]

            id+=1

    if not use_opencv:
        kernel = np.ones((8, 8), np.uint8)
        new_img = cv2.dilate(new_img, kernel, iterations = 1)
    else:
        kernel = np.ones((3, 3), np.uint8)
        new_img = cv2.dilate(new_img, kernel, iterations = 1)        
    new_img = cv2.resize(new_img,(resize_width,resize_height))

    return new_img


def undistort_process_by_img(ori_image, target_size, intrins, distort_params, target_intrins):
    ori_height, ori_width, _ = ori_image.shape
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrins, distort_params, np.eye(3), 
                                                     target_intrins, (ori_width, ori_height), cv2.CV_16SC2)
    undist = cv2.remap(ori_image, map1, map2, interpolation=cv2.INTER_LINEAR)
    new_img = cv2.resize(undist, target_size)

    return new_img