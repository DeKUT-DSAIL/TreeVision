import cv2
import numpy as np


def save_coefficients(path, *args):
    """
    Saves the camera matrix and distortion coefficients to a given path/file after single camera calibration
    @param path: The filepath where coefficients are saved
    @param args: The individual single camera coefficients/parameters to be saved
    """
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

    keys = ['K', 'D']
    for i in range(len(args)):
        file.write(f"{keys[i]}", args[i])
    
    file.release()



def load_coefficients(path):
    """
    Loads camera matrix and distortion coefficients from file
    @param path: The filepath where the coefficients are stored
    """
    coeff_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    mtx = coeff_file.getNode("K").mat()
    dist = coeff_file.getNode("D").mat()
    coeff_file.release()
    return [mtx, dist]



def save_stereo_coefficients(path, *args):
    """
    Save the coefficcients of the stereo system to file
    @param path: The filepath where all stereo coefficients will be saved
    @param args: The individual stereo coefficients/parameters to be saved
    """
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    
    keys = ['K1', 'D1', 'K2', 'D2', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']
    for i in range(len(args)):
        file.write(f"{keys[i]}", args[i])
    
    file.release()




def load_stereo_coefficients(path):
    """
    Loads stereo coefficients from file
    """
    # read file from storage
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    keys = ["K1", "D1", "K2", "D2", "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q"]
    [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q] = [file.getNode(key).mat() for key in keys]
    file.release()
    
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]



def projection_error(objpoints, imgpoints, tvecs, rvecs, mtx, dist):
    """
    This function computes the backprojection error in order to estimate the accuracy of the parameters found during
    calibration

    @param objpoints: Object points in the real world
    @param imgpoints: Image points (coordinates of points in the image plane)
    @param tvecs: Translation vector obtained during calibration
    @param rvecs: Rotation matrix obtained during calibration
    @param mtx: Camera matrix obtained during calibration
    @param dist: Camera distortion matrix obtained during calibration
    """
    mean_error = 0
    image_errors = []
    x = []
    y = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        error2 = [np.subtract(imgpoints2[j][0], imgpoints[i][j][0]) for j in range(len(imgpoints2))]

        image_errors.append(error)
        x.append([pair[0] for pair in error2])
        y.append([pair[1] for pair in error2])

        mean_error += error
    
    ME = mean_error/len(objpoints)

    return {'ME' :ME, 'Image errors': image_errors, 'X': x, 'Y': y}