import cv2
import numpy as np
import os
import argparse
from util.fisheye_undistortion_process import undistort_process_by_point as undistort_point
from util.fisheye_undistortion_process import undistort_process_by_img as undistort_img

parser = argparse.ArgumentParser(description='Fisheye undistortion project.')

parser.add_argument('--image-path', default='exp/demo_imgs/ori_fisheye_img.jpg', type=str)
parser.add_argument('--target-img-size', default=(960,640), type=int, nargs='+')
parser.add_argument('--save_root', default='exp/results', type=str)
parser.add_argument('--undistort-by-point', default=True, type=bool, help='Use point by point undistort method')
parser.add_argument('--use-opencv', dest='use_opencv', action='store_true', help='Use opencv lib')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the image using OpenCV
    image = cv2.imread(args.image_path)

    # Camera Params.
    intrins = np.array(
        [[417.756012, 0., 957.984985],
        [0., 418.014008, 642.856018],
        [0., 0., 1.]]
    )
    distort_params = np.array([0.175831, -0.013463, -0.016808, 0.0045])

    # Custom params.
    target_img_size = args.target_img_size
    result_root = args.save_root

    # whether to undistort by point.
    undistort_by_point = args.undistort_by_point

    # whether to use opencv lib.
    use_opencv = args.use_opencv 

    if undistort_by_point:
        undistorted_img = undistort_point(image, target_img_size, intrins, distort_params, use_opencv)
        if not use_opencv:
            cv2.imwrite(os.path.join(result_root, 'self_undistort.jpg'), undistorted_img)
        else:   
            cv2.imwrite(os.path.join(result_root, 'opencv_undistort.jpg'), undistorted_img)
    else:
        target_intrins = np.array(
            [[417.756012, 0., 957.984985],
            [0., 418.014008, 642.856018],
            [0., 0., 1.]]
        )
        undistorted_img = undistort_img(image, target_img_size, intrins, distort_params, target_intrins)
        cv2.imwrite(os.path.join(result_root, 'opencv_undistort_by_img.jpg'), undistorted_img)

    cv2.imshow("unditorted_image", undistorted_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
