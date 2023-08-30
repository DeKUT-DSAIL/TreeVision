import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def load_camera_params(config_file_path: str):
    '''
    Loads the camera parameters from the configuration file
    @param config_file_path: Path to the camera configuration file
    '''
    data = cv2.FileStorage(config_file_path, cv2.FILE_STORAGE_READ)
    keys = ["K1", "K2", "D1", "D2", "R1", "R2", "P1", "P2", "T", "Q"]
    [K1, K2, D1, D2, R1, R2, P1, P2, T, Q] = [data.getNode(key).mat() for key in keys]

    return [K1, K2, D1, D2, R1, R2, P1, P2, T, Q]



def rectify(image, config_file_path, side):
    '''
    Rectifies an image based on camera calibration parameters
    '''

    [K1, K2, D1, D2, R1, R2, P1, P2, T, _] = load_camera_params(config_file_path)

    '''
    We know that

            |f  0   cx1  0|                             |f  0   cx2  Tx*f|
    P1 =    |0  f   cy   0|         and         P2  =   |0  f   cy   0   |
            |0  f   1    0|                             |0  f   1    0   |
            
    and in our case, cx1 = cx2 = cx
    '''

    f = K1[0,0]
    Tx = T[0,0]
    P1 = np.hstack((K1, np.array([[0],[0],[0]])))
    P2 = np.hstack((K2, np.array([[Tx*f],[0],[0]])))

    h1, w1 = image.shape

    if side == "left":
        xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w1,h1), cv2.CV_32FC1)
        rectified_image = cv2.remap(image, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    elif side == "right":
        xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w1,h1), cv2.CV_32FC1)
        rectified_image = cv2.remap(image, xmap2, ymap2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    
    return rectified_image




def compute_depth_map(imgL: np.ndarray, imgR: np.ndarray, mask: np.ndarray, rectified: bool, sel: np.ndarray, config_file_path: str, min_disp, num_disp, block_size, uniqueness_ratio, speckle_window_size, speckle_range, disp_max_diff):
    '''
    This function extracts the disparity map from left and right images of a stereo image pair.
    @param imgL The left image of the stereo pair
    @param imgR The right image of the stereo pair
    @param mask The segmentation mask to be applied to imgL
    @param sel The structuring element or kernel to be applied to the mask. A vertical kernel might be good for masks of tree trunks and circular kernel better for tree crowns.
    '''

    # ------------------------------------- #
    # FILTER
    # ------------------------------------- #
    imgL = cv2.GaussianBlur(imgL, (5,5), 0)
    imgR = cv2.GaussianBlur(imgR, (5,5), 0)

    # ------------------------------------- #
    # STEREO RECTIFICATION
    # ------------------------------------- #
    if not rectified:

        imgL_rectified = rectify(imgL, config_file_path, "left")
        imgR_rectified = rectify(imgR, config_file_path, "right")
        mask_rectified = rectify(mask, config_file_path, "left")
    
        mask_rectified = morphology(mask_rectified, sel)
    
    else:
        mask_rectified = morphology(mask, sel)
        imgL_rectified = imgL
        imgR_rectified = imgR

    # -------------------------------- #
    # COMPUTE DISPARITY MAP
    # -------------------------------- #

    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = block_size,
            uniquenessRatio = uniqueness_ratio,
            speckleWindowSize = speckle_window_size,
            speckleRange = speckle_range,
            disp12MaxDiff = disp_max_diff,
            P1 = 8*3*block_size**2,
            P2 = 32*3*block_size**2)
    
    disp = stereo.compute(imgL_rectified, imgR_rectified)
    original = disp.copy()
    original = (original / 16).astype(np.float32)
    original = cv2.bitwise_and(original, original, mask=mask_rectified)
    
    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
    kernel= np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
    disp_closed = (closing - closing.min()) * 255
    disp_closed = disp_closed.astype(np.uint8)
    full = disp_closed.copy()
    disp_closed = cv2.bitwise_and(disp_closed, disp_closed, mask=mask_rectified)
    
    # R - Raw (before filtering);  F - Full (before masking); ;  O - Original
    return {'R': disp_closed, 'F': full, 'O': original}



def extract(left_im, right_im, mask, rectified, sel, config_file_path, min_disp, num_disp, block_size, uniqueness_ratio, speckle_window_size, speckle_range, disp_max_diff):
    '''
    Extracts the disparity map and returns it. The arguments min_disp - disp_max_diff are taken from OpenCV's SGBM_create(). 
    Visit https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html#adb7a50ef5f200ad9559e9b0e976cfa59 for details

    @param left_im: The left image
    @param right_im: The right image
    @param mask: The segmentation mask of the image used as the base for the disparity map (usually the left image)
    @param sel: The structuring element to be used for morphological processing
    @param config_file_path: Path to the camera configuration file, usually a YML file
    @param min_disp: See OpenCV SGBM_create() for details
    @param num_disp: See OpenCV SGBM_create() for details
    @param block_size: See OpenCV SGBM_create() for details
    @param uniqueness_ratio: See OpenCV SGBM_create() for details
    @param speckle_window_size: See OpenCV SGBM_create() for details
    @param speckle_range_size: See OpenCV SGBM_create() for details
    @param disp_max_diff: See OpenCV SGBM_create() for details
    '''
    dmap = compute_depth_map(
        imgL = left_im,
        imgR = right_im,
        mask = mask,
        rectified = rectified,
        sel = sel,
        config_file_path = config_file_path,
        min_disp = min_disp,
        num_disp = num_disp,
        block_size = block_size,
        uniqueness_ratio = uniqueness_ratio,
        speckle_window_size = speckle_window_size,
        speckle_range = speckle_range,
        disp_max_diff = disp_max_diff
    )

    depth = threshold_disparity(dmap['O'])

    return depth



def threshold_disparity(image):
    '''
    Performs two-level thresholding on the segmented disparity map. Some disparity maps have pixel intensities that exhibit \n
    a multi-modal histograms with most pixelintensities around the largest peak in the histogram. Values slightly below and \n
    slightly above this peak are considered backgrond pixels and are removed using this appraoch.

    @param image: Segmented disparity map
    '''

    mask = np.asarray(image > 10, dtype=np.uint8)

    hist = cv2.calcHist(images=[image], channels=[0], mask=mask, histSize=[256], ranges=[0,255])
    hist = np.squeeze(hist)

    peak_index = np.argmax(hist)

    thresh1 = np.where(hist == hist[peak_index])[0][0] - 10
    thresh2 = np.where(hist == hist[peak_index])[0][0] + 10

    thresholded = image.copy()
    thresholded[image < thresh1] = 0
    thresholded[image > thresh2] = 0

    return thresholded



def disp_to_dist(x):
    '''
    Resolves the distance from a given disparity value based on a curve fitted to the \n
    following vales:

    disp = [119, 109, 101, 94, 87, 82, 77, 74, 71, 65, 63, 60, 57, 55, 52, 50, 48, 47, 45, 
            43, 42, 41, 40, 39, 38, 37, 37, 35, 35, 33, 33, 31, 31]

    dist = np.linspace(3.0, 12.6, 33)

    @param x: The disparity value (usually the greyscale intensity of a pixel in the disparity map) to be resolved into distance in m
    '''
    y = (346*x**2 -116.7*x - 1.961) / (x**3 - 5.863*x**2 + 47.24*x + 487.6)
    Q = np.float32([[1,0,0,-640],[0,1,0,-360],[0,0,0,1442],[0,0,1/130,0]])
    _, _, y = cv2.reprojectImageTo3D(np.array([x], dtype=np.float32), Q)[0,0]
    y = y / 1000
    
    return y



def convex_hull(img):
    '''
    Finds edge pixels on the left, right, base and top of the image

    @param img: Source image, usually the segmented disparity map
    '''
    rows, cols = img.nonzero()
    base = (rows.max(), cols[np.where(rows == rows.max())][0])
    top =  (rows.min(), cols[np.where(rows == rows.min())][0])
    left =  (rows[np.where(cols == cols.min())][0], cols.min())
    right = (rows[np.where(cols == cols.max())][0], cols.max())
   
    return base, top, left, right



def median_top_pixel(image):
    '''
    Returns the median pixel intensity from the region of interest at the top of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    pixels = []
    top = convex_hull(image)[1]

    sub_image = image[top[0] : top[0] + 41, :]
    rows, columns = np.nonzero(sub_image)
    for row, column in zip(rows, columns):
        pixels.append(sub_image[row, column])
    
    pixels = np.array(pixels)
    med = np.median(pixels)

    return med



def median_base_pixel(image):
    '''
    Returns the median pixel intensity from the region of interest at the base of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    pixels = []
    base = convex_hull(image)[0]

    sub_image = image[base[0] - 50 : base[0] + 1, :]
    rows, columns = np.nonzero(sub_image)
    for row, column in zip(rows, columns):
        pixels.append(sub_image[row, column])

    pixels = np.array(pixels)
    med = np.median(pixels)
    image[base] = med

    return med



def median_bh_pixel(image, bh):
    '''
    Returns the median pixel intensity from the region of interest at the breast height of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    pixels = []
    sub_image = image[bh - 5 : bh + 5, :]

    rows, columns = np.nonzero(sub_image)
    for row, column in zip(rows, columns):
        pixels.append(sub_image[row, column])
    
    pixels = np.array(pixels)
    med = np.median(pixels)

    return med



def median_crown_pixel(image):
    '''
    Returns the median pixel intensity from the region of interest at the crown edges of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    left_pixels = []
    right_pixels = []

    left, right = convex_hull(image)[2:4]
    sub_image_left = image[:, left[1] : left[1] + 41]
    sub_image_right = image[:, right[1] - 40 : right[1] + 1]

    rows, columns = np.nonzero(sub_image_left)
    for row, column in zip(rows, columns):
        left_pixels.append(sub_image_left[row, column])

    rows, columns = np.nonzero(sub_image_right)
    for row, column in zip(rows, columns):
        right_pixels.append(sub_image_right[row, column])

    med_left = int(np.median(left_pixels))
    med_right = int(np.median(right_pixels))
    
    image[left] = med_left
    image[right] = med_right

    return med_left, med_right



def pixel_of_interest(image, parameter:str):
    '''
    Returns the value of the pixel of interest depending on the parameter chosen
    
    @param image The segmented disparity map
    @param parameter The parameter of interest. Can be "DBH", "CD" or "TH"
    '''
    if parameter.lower() == 'dbh':
        return median_base_pixel(image)
            
    elif parameter.lower() == 'th':
        base = median_base_pixel(image)
        top = median_top_pixel(image)
        return [base, top]
    
    elif parameter.lower() == 'cd':
        return median_crown_pixel(image)

    else:
        raise TypeError("Invalid parameter. Must be 'dbh', 'th', or 'cd' ")



def calculate_fields_of_view(dfov, width, height):
    '''
    Caculates the vertical and horizontal fields of view of the camera from the diagonal field
    of view. It is assumed that the left and right cameras have identical fields of view.
    @param dfov: The diagonal field of view of the camera in degrees
    @param width: Width of the camera image plane in pixels
    @param height: Height of the camera image plane in pixels
    '''
    dfov = np.deg2rad(dfov)

    diag_pixels = cv2.norm(np.array([height, width]))
    hfov = 2 * np.arctan(width * np.tan(dfov / 2) / diag_pixels)      
    vfov = 2 * np.arctan(height * np.tan(dfov / 2) / diag_pixels)

    return [hfov, vfov]



def morphology(mask, kernel):
    '''
    This function performs morphological closing followed by closing on the mask

    @param mask The image mask on which morphological processing is to be performed
    @param kernel The structuring element to be used for morphological processing
    '''
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return opening



def compute_bh(img, zc, baseline, focal_length, dfov, cx, cy):
    '''
    Computes the number of pixels from the trunk base breast height of the tree
    @param img: Source image
    @param zc: Real world depth of trunk base
    @param baseline: The stereo camera baseline in m
    @param focal_length: The focal length of the camera in pixels
    @param dfov: The diagonal field of view of the camera in degrees
    @param cx: The optical center of the camera image plane along the x axis
    @param cy: The optical center of the camera image plane along the y axis
    '''
    h, w = img.shape
    disparity = baseline * focal_length / zc
    base = convex_hull(img)[0]
    xc = baseline * (base[1] - cx) / disparity
    yc = baseline * (base[0] - cy) / disparity

    # applying the geometry for deriving the position of the breast height
    dg = cv2.norm(np.array([xc, yc, zc]))
    phi = np.arctan(yc / zc)
    beta = (np.pi/2 - phi)
    dh = np.sqrt(1.69 + dg**2 - 2.6*dg*np.cos(beta))
    theta = np.arcsin(1.3 * np.sin(beta) / dh)
    print(f"Angle subtended by breast height at camera: {round(np.rad2deg(theta), 2)} degrees")
    
    _, vfov = calculate_fields_of_view(dfov, w, h)
    sh = (h / np.tan(vfov / 2)) * np.tan(theta / 2) # no. of pixels from base to the breast height (1.3m above the ground)
    print(f"Distance in pixels from base to breast height: {sh}")
    bh = base[0] - np.int64(sh) # row number where breast height is found
    
    return bh



def compute_dbh(image, mask, baseline, focal_length, dfov, cx, cy):
    '''
    Extracts the DBH from the segmented disparity map
    @param image: The segmented disparity map
    @param mask: The segmentation mask
    @param dfov: The diagonal field of view of the camera in degrees
    '''
    
    base_px = pixel_of_interest(image, 'DBH')
    base_depth = disp_to_dist(base_px)
    print(f"Trunk base depth: {round(base_depth, 2)}m")

    bh = compute_bh(image, base_depth, baseline, focal_length, dfov, cx, cy)
    print(f"Breast Height Location: {bh} pixels from the top")

    bh_pixels = np.nonzero(mask[bh, :])[0]
    xmax = bh_pixels.max() # left_edge = (bh, xmin)
    xmin = bh_pixels.min() # right_edge = (bh, xmax)
    sd = xmax - xmin
    print(f"The DBH spans {sd} pixels")

    bh_px = median_bh_pixel(image=image, bh=bh)
    za = disp_to_dist(bh_px)
    zb = za
    print(f"Depth of breast height: {round(za, 2)}m")

    # Coordinates of left edge of the breast height
    left_edge_disparity = baseline * focal_length / za
    xa = baseline * (xmin - cx) / left_edge_disparity
    ya = baseline * (bh - cy) / left_edge_disparity

    # Coordinates of right edge of the breast height
    right_edge_disparity = baseline * focal_length / zb
    xb = baseline * (xmax - cx) / right_edge_disparity
    yb = baseline * (bh - cy) / right_edge_disparity

    print(f"Left BH edge coordinates: {round(xa, 2), round(ya, 2), round(za, 2)}")
    print(f"Right BH edge coordinates: {round(xb, 2), round(yb, 2), round(zb, 2)}")

    visible_dbh = xb - xa
    tangent_a = cv2.norm(np.array([xa, ya, za]))
    actual_dbh = visible_dbh * tangent_a / (np.sqrt(tangent_a ** 2 - (visible_dbh / 2) ** 2))
    print(f"Tangent A: {round(tangent_a, 2)}, \tTangent B: {round(cv2.norm(np.array([xb, yb, zb])), 2)}")
    print(f"Other DBH: {actual_dbh}")

    h, w = image.shape
    hfov, _ = calculate_fields_of_view(dfov, w, h)
    theta = np.arctan(sd * (np.tan(hfov / 2) / w))
    print(f"Angle subtended by trunk width at camera: {round(np.rad2deg(theta), 2)} degrees.")
    D = 2 * za * np.tan(theta)
    return D



def compute_cd(image, baseline, focal_length, dfov, cx, cy):
    '''
    This function extracts the crown diameter (CD) from a segmented depth map

    @param image: The segmented disparity map
    @param baseline: The stereo camera baseline in m
    @param focal_length: The focal length of the camera in pixels
    @param dfov: The diagonal field of view of the camera in degrees
    @param cx: The optical center of the camera image plane along the x axis
    @param cy: The optical center of the camera image plane along the y axis
    '''

    left, right = convex_hull(image)[2:4]
    left_crown_px, right_crown_px = pixel_of_interest(image, 'CD')
    
    z1 = disp_to_dist(left_crown_px)
    left_edge_disparity = baseline * focal_length / z1
    x1 = baseline * (left[1] - cx) / left_edge_disparity
    y1 = baseline * (left[0] - cy) / left_edge_disparity

    z2 = disp_to_dist(right_crown_px)
    right_edge_disparity = baseline * focal_length / z2
    x2 = baseline * (right[1] - cx) / right_edge_disparity
    y2 = baseline * (right[0] - cy) / right_edge_disparity

    print(f"Left crown edge coordinates: {round(x1, 2), round(y1, 2), round(z1, 2)}")
    print(f"Right crown edge coordinates: {round(x2, 2), round(y2, 2), round(z2, 2)}")

    sc = right[1] - left[1]
    print(f"Crown spans {sc} pixels")
    da = cv2.norm(np.array([x1, y1, z1]))

    h, w = image.shape
    hfov, _ = calculate_fields_of_view(dfov, w, h)
    theta = 2 * np.arctan(sc * (np.tan(hfov / 2) / w))
    CD = 2 * da * np.tan(theta/2)

    print(f"Left crown extreme is {round(da, 2)}m away")
    print(f"CD: {round(CD, 2)}m")
    print(f"Direct CD: {round(abs(x2 - x1), 2)}m")

    return CD



def compute_th(image, baseline, focal_length, dfov, cx, cy):
    '''
    This function extracts the tree height (TH) from a segmented depth map

    @param image The segmented disparity map
    @param baseline The stereo camera baseline in m
    @param f The focal length of the camera in pixels
    '''

    base, top = convex_hull(image)[0:2]
    base_px, top_px = pixel_of_interest(image, 'TH')
    print(f"Base: {base_px}")
    
    zb = disp_to_dist(base_px)
    base_disparity = baseline * focal_length / zb
    xb = baseline * (base[1] - cx) / base_disparity
    yb = baseline * (base[0] - cy) / base_disparity
    print(f"Base coordinates: {round(xb, 2), round(yb, 2), round(zb, 2)}")

    zt = disp_to_dist(top_px)
    top_disparity = baseline * focal_length / zt
    xt = baseline * (top[1] - cx) / top_disparity
    yt = baseline * (top[0] - cy) / top_disparity
    print(f"Top coordinates: {round(xt, 2), round(yt, 2), round(zt, 2)}")    

    st = base[0] - top[0]
    db = cv2.norm(np.array([xb, yb, zb]))
    dt = cv2.norm(np.array([xt, yt, zt]))
    phi = np.arctan(zb / yb)
    h, w = image.shape
    _, vfov = calculate_fields_of_view(dfov, w, h)
    theta = 2 * np.arctan(st * (np.tan(vfov / 2) / h))
    TH = np.abs(dt * np.sin(theta) / np.sin(phi))

    print(f"Tree base is {round(db, 2)}m away")
    print(f"Tree top is {round(dt, 2)}m away")
    print(f"TH: {round(TH, 2)}m")
    print(f"Direct TH: {round(abs(yt - yb), 2)}m")

    return TH