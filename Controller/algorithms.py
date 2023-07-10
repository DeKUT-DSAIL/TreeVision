import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def compute_depth_map(imgL: np.ndarray, imgR: np.ndarray, mask: np.ndarray, sel: np.ndarray, config_file_path: str, min_disp, num_disp, block_size, uniqueness_ratio, speckle_window_size, speckle_range, disp_max_diff):
    '''
    This function extracts the disparity map from left and right images of a stereo image pair. \n
    @param imgL The left image of the stereo pair \n
    @param imgR The right image of the stereo pair \n
    @param mask The segmentation mask to be applied to imgL \n
    @param sel The structuring element or kernel to be applied to the mask. A vertical kernel might be good for masks of tree trunks and circular kernel better for tree crowns. \n
    '''

    # ------------------------------------- #
    # SETUP
    # ------------------------------------- #
    imgL = cv2.GaussianBlur(imgL, (5,5), 0)
    imgR = cv2.GaussianBlur(imgR, (5,5), 0)

    # read camera data
    data = cv2.FileStorage(config_file_path, cv2.FILE_STORAGE_READ)
    keys = ["K1", "K2", "D1", "D2", "R1", "R2", "P1", "P2", "T"]
    [K1, K2, D1, D2, R1, R2, P1, P2, T] = [data.getNode(key).mat() for key in keys]

    '''
    We know that

            |f  0   cx1  0|
    P1 =    |0  f   cy   0|
            |0  f   1    0|

    and 

            |f  0   cx2  Tx*f|
    P2 =    |0  f   cy   0   |
            |0  f   1    0   |

    and in our case, cx1 = cx2 = cx
    '''

    f = K1[0,0]
    Tx = T[0,0]
    P1 = np.hstack((K1, np.array([[0],[0],[0]])))
    P2 = np.hstack((K2, np.array([[Tx*f],[0],[0]])))


    # ------------------------------------- #
    # STEREO RECTIFICATION
    # ------------------------------------- #
    h1, w1 = imgL.shape

    # rectify images using initUndistortRectifyMap
    xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w1,h1), cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w1,h1), cv2.CV_32FC1)

    imgL_rectified = cv2.remap(imgL, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    imgR_rectified = cv2.remap(imgR, xmap2, ymap2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    mask_rectified = cv2.remap(mask, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    mask_rectified = morphology(mask_rectified, sel)

    # -------------------------------- #
    # COMPUTE DISPARITY MAP
    # -------------------------------- #

    # Matched blocked size
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
    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

    kernel= np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
    disp_closed = (closing - closing.min()) * 255
    disp_closed = disp_closed.astype(np.uint8)
    full = disp_closed
    disp_closed[mask_rectified == 0] = 0
    
    # R - Raw (before filtering);  F - Filtered;  O - Full (before masking)
    return {'R': disp_closed, 'O': full}



def extract(left_im, right_im, mask, sel, config_file_path, min_disp, num_disp, block_size, uniqueness_ratio, speckle_window_size, speckle_range, disp_max_diff):
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

    depth = threshold_disparity(dmap['R'])

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

    thresh1 = np.where(hist == hist[peak_index])[0][0] - 20
    thresh2 = np.where(hist == hist[peak_index])[0][0] + 20

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
    
    # print(pixels)
    pixels = np.array(pixels)

    return np.median(pixels)



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

    # print(pixels)
    pixels = np.array(pixels)

    return np.median(pixels)



def median_bh_pixels(image):
    '''
    Returns the median pixel intensity from the region of interest at the breast height of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    pixels = []
    zc = disp_to_dist(median_base_pixel(image))
    bh = compute_bh(image, zc)
    sub_image = image[bh - 5 : bh + 5, :]

    rows, columns = np.nonzero(sub_image)
    for row, column in zip(rows, columns):
        pixels.append(sub_image[row, column])
    
    pixels = np.array(pixels)
    half_pixels = np.array_split(pixels, 2)
    center = np.median(pixels)
    edge = np.median(half_pixels[0])

    return [center, edge]



def median_crown_pixel(image):
    '''
    Returns the median pixel intensity from the region of interest at the crown edges of the tree in the disparity map

    @param image: Source image, usually the segmented disparity map
    '''

    pixels = []
    left = convex_hull(image)[2]

    sub_image = image[:, left[1] : left[1] + 41]
    rows, columns = np.nonzero(sub_image)
    for row, column in zip(rows, columns):
        pixels.append(sub_image[row, column])

    # print(pixels)
    pixels = np.array(pixels)

    return int(np.median(pixels))



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
    


def morphology(mask, kernel):
    '''
    This function performs morphological closing followed by closing on the mask

    @param mask The image mask on which morphological processing is to be performed
    @param kernel The structuring element to be used for morphological processing
    '''
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return opening



def compute_bh(img, zc, baseline=0.129, f=1438):
    '''
    Computes the number of pixels from the trunk base breast height of the tree
    @param img: Source image
    @param zc: Real world depth of trunk base
    '''
    disparity = baseline * f / zc
    base = convex_hull(img)[0]
    xc = baseline * (base[0] - 360) / disparity
    yc = baseline * (base[1] - 640) / disparity

    # applying the geometry for deriving the position of the breast height
    dg = np.sqrt(xc**2 + yc**2 + zc**2)
    phi = np.arctan(yc / zc)
    beta = (np.pi/2 - phi)
    dh = np.sqrt(1.69 + dg**2 - 2.6*dg*np.cos(beta))
    theta = np.arcsin(1.3 * np.sin(beta) / dh)
    print(f"Angle subtended by breast height at camera: {round(theta*57.2958, 2)} degrees")
    sh = 2822.61 * np.tan(theta/2) # no. of pixels from base to the breast height (1.3m above the ground)
    bh = base[0] - np.int64(sh) # row number where breast height is found
    
    return bh



def compute_dbh(image, mask):
    '''
    Extracts the DBH from the segmented disparity map
    @param image: The segmented disparity map
    @param sd: The number of pixels spanned by the image of the tree trunk at the breast height
    @param da: The real world distance between the camera and the tree trunk along the tangent linking the camera and the circumference of the tree trunk at the breast heght
    '''
    
    base_px = pixel_of_interest(image, 'DBH')
    base_depth = disp_to_dist(base_px)
    print(f"Trunk base depth: {round(base_depth, 2)}m")

    bh = compute_bh(img=image, zc=base_depth)
    print(f"Breast Height Location: {bh} pixels from the top")

    bh_pixels = np.nonzero(mask[bh, :])[0]
    sd = bh_pixels.size
    print(f"The DBH spans {sd} pixels")

    da = disp_to_dist(median_bh_pixels(image)[0])
    print(f"Depth of breast height: {round(da, 2)}m")

    theta = np.arctan(sd * 3.546e-4)
    print(f"Angle subtended by trunk width at camera: {round(theta*57.2958, 2)} degrees.")
    D = 2 * da * np.tan(theta)
    return D



def compute_cd(image, baseline=0.129, f=1438):
    '''
    This function extracts the crown diameter (CD) from a segmented depth map

    @param image: The segmented disparity map
    @param baseline: The stereo camera baseline in m
    @param f: The focal length of the camera in pixels
    '''

    left, right = convex_hull(image)[2:4]
    crown_px = pixel_of_interest(image, 'CD')
    print(f"Crown edge pixel intensity: {crown_px}")
    
    z = disp_to_dist(crown_px)
    disparity = baseline * f / z
    x = baseline * (left[0] - 360) / disparity
    y = baseline * (left[1] - 640) / disparity
    print(f"Real word coordinates: {round(x, 2), round(y, 2), round(z, 2)}")

    sc = right[1] - left[1]
    print(f"Crown spans {sc} pixels")
    da = np.sqrt(x**2 + y**2 + z**2)
    theta = 2 * np.arctan(sc * 3.546e-4)
    CD = 2 * da * np.tan(theta/2)

    print(f"Left crown extreme is {round(da, 2)}m away")
    print(f"CD: {round(CD, 2)}m")

    return CD



def compute_th(image, baseline=0.129, f=1438):
    '''
    This function extracts the tree height (TH) from a segmented depth map

    @param image The segmented disparity map
    @param baseline The stereo camera baseline in m
    @param f The focal length of the camera in pixels
    '''

    base, top = convex_hull(image)[0:2]
    base_px, top_px = pixel_of_interest(image, 'TH')
    
    zb = disp_to_dist(base_px)
    disparity = baseline * f / zb
    xb = baseline * (base[0] - 360) / disparity
    yb = baseline * (base[1] - 640) / disparity
    print(f"Real world coordinates of base: {round(xb, 2), round(yb, 2), round(zb, 2)}")

    zt = disp_to_dist(top_px)
    disparity = baseline * f / zt
    xt = baseline * (top[0] - 360) / disparity
    yt = baseline * (top[1] - 640) / disparity
    print(f"Real world coordinates of top: {round(xt, 2), round(yt, 2), round(zt, 2)}")    

    st = base[0] - top[0]
    db = np.sqrt(xb**2 + yb**2 + zb**2)
    dt = np.sqrt(xt**2 + yt**2 + zt**2)
    phi = np.arctan(zb / yb)
    theta = 2 * np.arctan(st * 3.543e-4)
    TH = np.abs(dt * np.sin(theta) / np.sin(phi))

    print(f"Tree base is {round(db, 2)}m away")
    print(f"Tree top is {round(dt, 2)}m away")
    print(f"TH: {round(TH, 2)}m")

    return TH