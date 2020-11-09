import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # image pixel grid

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob("camera_cal/calibration*.jpg")

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.waitKey(100)


def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])  # x, y
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


import matplotlib.image as mpimg


def hls_color_thd(img, threshold=(0, 255), color_opt=("hls")):
    img_in = np.copy(img)

    if color_opt == "hls":
        hls = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= threshold[0]) & (s_channel <= threshold[1])] = 1

        return s_binary

    elif color_opt == "rgb":
        rgb = img_in
        r_channel = rgb[:, :, 0]
        g_channel = rgb[:, :, 1]
        b_channel = rgb[:, :, 2]

        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= threshold[0]) & (r_channel <= threshold[1])] = 1

        return r_binary

    else:
        print("set the color space")
        return np.zeros_like(img_in)


def gradient_thd(img, sobel_kernel=3, threshold=(0, 255), manner_opt=("mag")):
    img_in = np.copy(img)

    gray = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # x dir
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # y dir

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    scaled_sobelx = np.uint8(
        255 * abs_sobelx / np.max(abs_sobelx)
    )  # normalized is needed to adapt threshold consistence
    scaled_sobely = np.uint8(
        255 * abs_sobely / np.max(abs_sobely)
    )  # normalized is needed to adapt threshold consistence

    sobel_xy = np.sqrt(sobelx ** 2 + sobely ** 2)  # maginitude
    scaled_sobelxy = np.uint8(
        255 * sobel_xy / np.max(sobel_xy)
    )  # normalized is needed to adapt threshold consistence

    direction = np.arctan2(abs_sobely, abs_sobelx)  # direction

    if manner_opt == "mag":
        binary_mag = np.zeros_like(scaled_sobelxy)
        binary_mag[
            (scaled_sobelxy >= threshold[0]) & (scaled_sobelxy <= threshold[1])
        ] = 1
        return binary_mag

    elif manner_opt == "absx":
        binary_absx = np.zeros_like(scaled_sobelx)
        binary_absx[
            (scaled_sobelx >= threshold[0]) & (scaled_sobelx <= threshold[1])
        ] = 1
        return binary_absx

    elif manner_opt == "absy":
        binary_absy = np.zeros_like(scaled_sobely)
        binary_absy[
            (scaled_sobely >= threshold[0]) & (scaled_sobely <= threshold[1])
        ] = 1

        return binary_absy

    elif manner_opt == "dir":
        binary_dir = np.zeros_like(direction)
        binary_dir[(direction >= threshold[0]) & (direction <= threshold[1])] = 1

        return binary_dir

    else:
        print("set the gradient manner")


def perspective_img(undist_img, src_pts):
    # src to dst
    # set the four-pointt (warp rectangular, ROI)
    img_in = np.copy(undist_img)

    undistorted = undist_img

    offset = 50
    dst = np.float32(
        [
            [offset, offset],
            [img_in.shape[1] - offset, offset],
            [img_in.shape[1] - offset, img_in.shape[0] - offset],
            [offset, img_in.shape[0] - offset],
        ]
    )

    M = cv2.getPerspectiveTransform(src_pts, dst)
    warped = cv2.warpPerspective(undistorted, M, (img.shape[1], img.shape[0]))

    pts = np.array(src_pts, np.int32)
    pts = pts.reshape(-1, 2)
    in_roi = np.copy(undistorted)
    roi = cv2.polylines(
        in_roi, [pts], True, (1, 1, 1), 3
    )  ########### comment out, for visualization (11/3)

    return roi, warped, M


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    except TypeError:
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, init_tune):
    # HYPER PARAMETER
    margin = 150

    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    left_lane_inds = (
        nonzerox
        > (
            init_tune[0][0] * (nonzeroy) ** 2
            + init_tune[0][1] * nonzeroy
            + init_tune[0][2]
            - margin
        )
    ) & (
        nonzerox
        < (
            init_tune[0][0] * (nonzeroy) ** 2
            + init_tune[0][1] * nonzeroy
            + init_tune[0][2]
            + margin
        )
    )
    right_lane_inds = (
        nonzerox
        > (
            init_tune[1][0] * (nonzeroy) ** 2
            + init_tune[1][1] * nonzeroy
            + init_tune[1][2]
            - margin
        )
    ) & (
        nonzerox
        < (
            init_tune[1][0] * (nonzeroy) ** 2
            + init_tune[1][1] * nonzeroy
            + init_tune[1][2]
            + margin
        )
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_poly(
        binary_warped.shape, leftx, lefty, rightx, righty
    )

    out_img = np.copy(binary_warped)

    return out_img, left_fitx, right_fitx, ploty


def measure_curvature(left_fitx, right_fitx, ploty, ratio=(1, 1)):
    img_x_size = 1280  # image x size

    left_fit_cr = np.polyfit(ploty * ratio[1], left_fitx * ratio[0], 2)
    right_fit_cr = np.polyfit(ploty * ratio[1], right_fitx * ratio[0], 2)

    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = (
        1 + (2 * left_fit_cr[0] * y_eval * ratio[1] + left_fit_cr[1]) ** 2
    ) ** 1.5 / np.absolute(2 * left_fit_cr[0])
    right_curverad = (
        1 + (2 * right_fit_cr[0] * y_eval * ratio[1] + right_fit_cr[1]) ** 2
    ) ** 1.5 / np.absolute(2 * right_fit_cr[0])

    mean_curverad = np.mean([left_curverad, right_curverad])

    left_x = (
        left_fit_cr[0] * (y_eval * ratio[1]) ** 2
        + left_fit_cr[1] * (y_eval * ratio[1])
        + left_fit_cr[2]
    )
    left_of_center = (img_x_size / 2) * ratio[0] - left_x

    return left_curverad, right_curverad, mean_curverad, left_of_center


def process_img(image):
    process_img.running_flag += 1

    img = np.copy(image)
    # FLOW

    # step 1: undistortion
    undist = cal_undistort(img, objpoints, imgpoints)

    # step 2: thresholding (color, gradient, combination)
    color_thd = hls_color_thd(undist, threshold=(180, 255), color_opt="hls")
    gradient_thd_x = gradient_thd(
        undist, sobel_kernel=3, threshold=(20, 100), manner_opt="absx"
    )
    gradient_thd_y = gradient_thd(
        undist, sobel_kernel=3, threshold=(20, 100), manner_opt="absy"
    )
    gradient_thd_mag = gradient_thd(
        undist, sobel_kernel=3, threshold=(20, 100), manner_opt="mag"
    )
    gradient_thd_dir = gradient_thd(
        undist,
        sobel_kernel=3,
        threshold=(0 / 180 * np.pi, 60 / 180 * np.pi),
        manner_opt="dir",
    )
    comb_bin = np.zeros_like(gradient_thd_mag)
    comb_bin[
        ((color_thd == 1))
        | ((gradient_thd_x == 1))
        | ((gradient_thd_mag == 1) & (gradient_thd_dir == 1) & (gradient_thd_y == 1))
    ] = 1

    # step 3: perspective
    src = np.float32([[490, 515], [835, 515], [1080, 650], [265, 650]])
    roi, warped, M = perspective_img(comb_bin, src)

    # step 4: Search from Prior
    left_fit = np.array([8.22279110e-05, -8.01574626e-02, 1.80496286e02])
    right_fit = np.array([9.49537809e-05, -9.58782039e-02, 1.18196061e03])
    init_tune = np.array([left_fit, right_fit])

    result, left_fitx, right_fitx, ploty = search_around_poly(warped, init_tune)

    # step 5: curvature
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 1000  # meters per pixel in x dimension
    ratio = [xm_per_pix, ym_per_pix]

    left_curverad, right_curverad, mean_curverad, left_of_center = measure_curvature(
        left_fitx, right_fitx, ploty, ratio
    )
    str_curv = (
        "Radius of Curvature = %6d" % mean_curverad
        + "(m)     Vehicle is %.2f" % left_of_center
        + "m left of center"
    )

    # step 6: Inverse Warp
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(result).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    offset = 50
    dst = np.float32(
        [
            [offset, offset],
            [result.shape[1] - offset, offset],
            [result.shape[1] - offset, result.shape[0] - offset],
            [offset, result.shape[0] - offset],
        ]
    )

    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(
        color_warp, (Minv), (result.shape[1], result.shape[0])
    )

    # Combine the result with the original image
    fin_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(
        fin_img,
        str_curv,
        (10, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
    )

    fin = fin_img

    return fin


from moviepy.editor import VideoFileClip
from IPython.display import HTML

videos = glob.glob("*.mp4")
for ea in videos:

    process_img.running_flag = 0
    src_string = ea
    src_video = VideoFileClip(src_string)
    out_video = "output_videos/out_" + src_string

    img_clip = src_video.fl_image(process_img)
    img_clip.write_videofile(out_video, audio=False)