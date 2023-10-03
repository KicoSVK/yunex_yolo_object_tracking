import cv2
import cv2 as cv
import numpy as np
import os
import glob

def getHomography(points_camX, points_camRef, H_camX, use_known_points=True):
    if use_known_points:
        H_camX, _ = cv2.findHomography(points_camX, points_camRef, cv2.RANSAC, 3)
        return H_camX

    print("Now, automatic point extraction from image is not supported.")
    return None

def getIntrinsicParameters(K, D):
    img_paths = glob.glob('chessboard/*.png')
    chess_size = (9, 6)
    img_size = None
    points_2D_vec = []
    points_3D_vec = []

    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_size = img.shape[::-1]

        chess_points_num = chess_size[0] * chess_size[1]

        success, points_2D = cv2.findChessboardCorners(img, chess_size)
        if success:
            points_2D_vec.append(points_2D)

            points_3D = np.zeros((chess_points_num, 3), np.float32)
            points_3D[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)
            points_3D_vec.append(points_3D)

    success_calib, K, D, rvecs, tvecs = cv2.calibrateCamera(points_3D_vec, points_2D_vec, img_size, None, None)

    # print(K)
    # print(D)

    # try undistort img for control
    # img_undist = cv2.undistort(img, K, D)
    # cv2.imshow('Undist', img_undist)
    # cv2.waitKey(0)

    return success_calib, K, D

def computeCameraParameters(object_points_3D, image_points_2D, image_points_2D_ref, img_size):
    K, D, R, H, P = None, None, None, None, None
    rvecs, tvecs = None, None

    #distortion_coeffs = np.zeros((4,1))
    #focal_length = size[1]
    #center = (size[1]/2, size[0]/2)
    #matrix_camera = np.array(
    #                     [[focal_length, 0, center[0]],
    #                     [0, focal_length, center[1]],
    #                     [0, 0, 1]], dtype = "double"
    #                     )

    object_points_3D_vec = [object_points_3D]
    image_points_2D_vec = [image_points_2D]

    #cv2.calibrateCamera(object_points_3D_vec, image_points_2D_vec, img_size, K, D, rvecs, tvecs)

    success_calib, K, D = getIntrinsicParameters(K, D)
    if success_calib:

        #success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0)
        _, rvecs, tvecs = cv2.solvePnP(np.array(object_points_3D_vec, dtype=np.float32),
                                        np.array(image_points_2D_vec, dtype=np.float32),
                                        K, D)
        #cv2.solvePnP(object_points_3D_vec[0], image_points_2D_vec[0], K, D, rvecs, tvecs, False)

        tempR = cv2.Rodrigues(rvecs)
        R = tempR[0]

        RT = np.array([[R[0, 0], R[0, 1], R[0, 2], tvecs[0, 0]],
               [R[1, 0], R[1, 1], R[1, 2], tvecs[1, 0]],
               [R[2, 0], R[2, 1], R[2, 2], tvecs[2, 0]]], dtype=np.float64)

        #print("RT: ", RT)

        # computation of projection matrix
        P = K @ RT

        # compute homographies which map camerasX points into camera reference (birdView) image plane
        H = getHomography(np.array(image_points_2D), np.array(image_points_2D_ref), H)
    else:
        print("Error, getIntrinsicParameters() failed.")

    return H, K, D, P

def getPointInRefPlane(point_camX, H_camX):
    point_camX_vec = [point_camX]
    point_camRef_vec = cv2.perspectiveTransform(
        np.array([point_camX_vec], dtype=np.float32), H_camX)
    return tuple(point_camRef_vec[0][0].astype(int))

def mytriangulate_points(projMatr1, projMatr2, projPoints1, projPoints2):
    matr1 = np.asarray(projMatr1)
    matr2 = np.asarray(projMatr2)
    points1 = np.asarray(projPoints1)
    points2 = np.asarray(projPoints2)

    if points1.ndim == 2 and points1.shape[1] == 2:
        points1 = points1.reshape(-1, 1, 2)
    if points2.ndim == 2 and points2.shape[1] == 2:
        points2 = points2.reshape(-1, 1, 2)

    points4D = cv2.triangulatePoints(matr1, matr2, points1, points2)
    #points4D /= points4D[3]

    #return points4D[:3]
    return points4D

def triangulate_points(projMatr1, projMatr2, point_camX, point_camRef):
    if (
        not isinstance(projMatr1, np.ndarray)
        or not isinstance(projMatr2, np.ndarray)
        or not isinstance(point_camX, np.ndarray)
        or not isinstance(point_camRef, np.ndarray)
    ):
        raise ValueError("Input parameters must be NumPy arrays")

    if (
        projMatr1.shape != (3, 4)
        or projMatr2.shape != (3, 4)
        or point_camX.shape != (3,)
        or point_camRef.shape != (3,)
    ):
        raise ValueError("Input parameter shapes are not as expected")

    numPoints = 1  # Since you have a single point
    points4D = np.zeros((4, numPoints), dtype=np.float64)

    for i in range(numPoints):
        matrA = np.zeros((4, 4), dtype=np.float64)

        for j in range(2):
            x = point_camX[0] if j == 0 else point_camRef[0]
            y = point_camX[1] if j == 0 else point_camRef[1]

            for k in range(4):
                matrA[j * 2 + 0, k] = x * projMatr1[2, k] - projMatr1[0, k]
                matrA[j * 2 + 1, k] = y * projMatr1[2, k] - projMatr1[1, k]

        _, _, matrV = np.linalg.svd(matrA)

        points4D[:, i] = matrV[:, -1]  # Store the 4D point

    return points4D


def getPositionXYZ(P_camX, P_camRef, point_camX, point_camRef):

    #print("P_camX")
    #print(P_camX)
    #print("P_camRef")
    #print(P_camRef)

    #print("point_camX")
    #print(point_camX)
    #print("point_camRef")
    #print(point_camRef)

    #point_camX = np.array(point_camX)
    #point_camRef = np.array(point_camRef)
    #P_camX = np.round(P_camX, decimals=2)
    #P_camRef = np.round(P_camRef, decimals=2)

    #P_camX = np.array([[-20.43243679012022, 853.9798347022981, 287.4819102677112, 5089.663215124147],
    #               [-19.66667252208214, -30.76094245134698, 623.8419230885621, 5012.478776710186],
    #               [0.667102527341207, 0.5979747533157282, 0.4442976619474479, 8.842163259354772]])

    #P_camRef = np.array([[-442.1431404434938, 72.39144679137908, 782.0574231299992, 17133.36394068422],
    #               [98.9978106950954, -538.0474051872916, 302.0201679362187, 8932.977842396387],
    #               [0.2870283902148107, 0.1290448157922935, 0.9491902542313777, 21.59996910789519]])

    #print("matrix:")
    #print(np.array(P_camX))
    #print(np.array(P_camRef))
    #print(np.array(point_camX))
    #print(np.array(point_camRef))
    #print((P_camX))
    #print((P_camRef))
    #print((point_camX))
    #print((point_camRef))

    #cv2.triangulatePoints(P_camX, P_camRef, point_camX, point_camRef, points4D)
    #######################points4D = mytriangulate_points(np.array(P_camX), np.array(P_camRef), np.array(point_camX), np.array(point_camRef))

    # Resulting 3d points
    #points3d = (points4D[:3, :]/points4D[3, :]).T
    #print(points3d)

    # Define the 2D points in homogeneous coordinates
    ######point_camX = np.array([point_camX[0], point_camX[1]])
    #######point_camRef = np.array([point_camRef[0], point_camRef[1]])

    # Create matrices for the 2D points
    #WORKINGGGGGGGGG#point_camX_vec = np.array([point_camX[0], point_camX[1], 1])
    #WORKINGGGGGGGGG#point_camRef_vec = np.array([point_camRef[0], point_camRef[1], 1])
    
    point_camX_vec = np.array([902, 332, 1])  # Assuming the third coordinate is 1 for 3D point
    point_camRef_vec = np.array([780, 178, 1])  # Assuming the third coordinate is 1 for 3D point
    #point_camX_vec = np.array([902, 332])
    #point_camRef_vec = np.array([780, 178])



    # Create a NumPy array to represent points4D
    points4D = np.empty((4, 1), dtype=np.float32)

    ###
    # Reshape the input vectors
    #point_camX_vec = point_camX_vec.reshape(1, -1)
    #point_camRef_vec = point_camRef_vec.reshape(1, -1)

    # Perform triangulation
    #points4D = cv2.triangulatePoints(P_camX, P_camRef, point_camX_vec.T, point_camRef_vec.T)

    # The result 'points4D' contains homogeneous coordinates, so you may want to normalize them
   # points3D = cv2.convertPointsFromHomogeneous(points4D.T)
    ###

    #print(" ")
    #print("P_camX")
    #print(P_camX)
    #print("P_camRef")
    #print(P_camRef)
    #print(" ")
    #print(point_camX_vec)
    #print(" ")
    #print(point_camRef_vec)

    # Perform triangulation
    #cv2.triangulatePoints(P_camX, P_camRef, point_camX_vec, point_camRef_vec, points4D)
    #points4D = cv2.triangulatePoints(P_camX, P_camRef, point_camX_vec.T, point_camRef_vec.T)
    ######points4D = triangulate_points(P_camX, P_camRef, point_camX_vec, point_camRef_vec)
    # Call the triangulate_points function with your inputs
    points4D = triangulate_points(P_camX, P_camRef, point_camX_vec, point_camRef_vec)

    print(" ")
    print("points4D")
    print(points4D)

    # Convert the homogeneous coordinates to 3D points
    #points3D = points4D[:3] / points4D[3]

    # Print the resulting 3D points
    #print("Triangulated 3D Points:")
    #for i in range(points3D.shape[1]):
    #    print(f"Point {i+1}: ({points3D[0, i]}, {points3D[1, i]}, {points3D[2, i]})")

    #print(" ")
    #print(" ")
    #print(" ")
    #print(np.array(P_camX))
    #print(" ")
    #print(np.array(P_camRef))
    #print(" ")
    #print("points4D:")
    #print(points4D)

    #XYZ_vec = []
    
    #for i in range(points4D.shape[1]):
    #    h = points4D[3, i]
    #    XYZ = np.zeros((3, 1))
    #    XYZ[0] = points4D[0, i] / h
    #    XYZ[1] = points4D[1, i] / h
    #    XYZ[2] = points4D[2, i] / h
    #    XYZ_vec.append(tuple(XYZ.squeeze().tolist()))

    #return XYZ_vec
