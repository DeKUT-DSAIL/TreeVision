import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def disp_to_dist(x):
    '''
    Resolves the distance from a given disparity value based on a curve fitted to the \n
    following vales:

    disp = [119, 109, 101, 94, 87, 82, 77, 74, 71, 65, 63, 60, 57, 55, 52, 50, 48, 47, 45, 
            43, 42, 41, 40, 39, 38, 37, 37, 35, 35, 33, 33, 31, 31]

    dist = np.linspace(3.0, 12.6, 33)
    '''
    y = (346*x**2 -116.7*x - 1.961) / (x**3 - 5.863*x**2 + 47.24*x + 487.6)
    return y



def convex_hull(img):
    '''
    Finds the tip pixels on the left, right, base and top of the image
    '''
    rows, cols = img.nonzero()
    base = (rows.max(), cols[np.where(rows == rows.max())][0])
    top =  (rows.min(), cols[np.where(rows == rows.min())][0])
    left =  (rows[np.where(cols == cols.min())][0], cols.min())
    right = (rows[np.where(cols == cols.max())][0], cols.max())
   
    return base, top, left, right



def compute_bh(img, zc, baseline=0.129, f=1438):
    '''
    Computes the number of pixels from the trunk base breast height of the tree
    @ img: Source image
    @ zc: Real world depth of trunk base
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



def compute_dbh(sd, da):
    theta = np.arctan(sd * 3.546e-4)
    print(f"Angle subtended by trunk width at camera: {round(theta*57.2958, 2)} degrees.")
    D = 2 * da * np.tan(theta)
    return D



def median_top_pixel(image):
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
    '''
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return opening



def compute_depth_map(imgL: np.ndarray, imgR: np.ndarray):
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
        data = cv2.FileStorage("/home/cedric/work/projects/Forest/configs/stereo.yml", cv2.FILE_STORAGE_READ)
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
        # mask_rectified = cv2.remap(mask, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # mask_rectified = morphology(mask_rectified, sel)

        # -------------------------------- #
        # COMPUTE DISPARITY MAP
        # -------------------------------- #

        # Matched blocked size
        window_size = 11
        min_disp = 0
        num_disp = 128-min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = window_size,
                uniquenessRatio = 10,
                speckleWindowSize = 100,
                speckleRange = 2,
                disp12MaxDiff = 5,
                P1 = 8*3*window_size**2,
                P2 = 32*3*window_size**2)
        
        # right_matcher = cv2.ximgproc.createRightMatcher(stereo)

        # WLS Filter Parameters
        lmbda = 8000
        sigma=1.4
        visual_multiplier = 1.0

        # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        # wls_filter.setLambda(lmbda)
        # wls_filter.setSigmaColor(sigma)

        # Compute the depth images
        disp = stereo.compute(imgL_rectified, imgR_rectified)
        dispL = disp
        # dispR = right_matcher.compute(imgR_rectified, imgL_rectified)
        dispL = np.int16(dispL)
        # dispR = np.int16(dispR)

        # Filtering with the WLS Filter
        # filtered = wls_filter.filter(dispL, imgL_rectified, None, dispR)
        # filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        # filtered = np.uint8(filtered)

        disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
        kernel= np.ones((3,3),np.uint8)
        # apply morphological closing to remove little black holes (removing noise)
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
        disp_closed = (closing - closing.min()) * 255
        disp_closed = disp_closed.astype(np.uint8)
        full = disp_closed

        # mask the disparity maps
        # filtered[mask_rectified == 0] = 0
        # disp_closed[mask_rectified == 0] = 0
        
        # R - Raw (before filtering);  F - Filtered;  O - Full (before masking)
        return {'R': disp_closed, 'O': full}