# '''
# Sample Usage:-
# python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
# '''


# import numpy as np
# import cv2
# import sys
# from utils import ARUCO_DICT
# import argparse
# import time


# def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

#     '''
#     frame - Frame from the video stream
#     matrix_coefficients - Intrinsic matrix of the calibrated camera
#     distortion_coefficients - Distortion coefficients associated with your camera

#     return:-
#     frame - The frame with the axis drawn on it
#     '''

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters_create()


#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
#         cameraMatrix=matrix_coefficients,
#         distCoeff=distortion_coefficients)

#         # If markers are detected
#     if len(corners) > 0:
#         for i in range(0, len(ids)):
#             # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
#             rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                        distortion_coefficients)
#             # Draw a square around the markers
#             cv2.aruco.drawDetectedMarkers(frame, corners) 

#             # Draw Axis
#             cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

#     return frame

# if __name__ == '__main__':

#     ap = argparse.ArgumentParser()
#     ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
#     ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
#     ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
#     args = vars(ap.parse_args())

    
#     if ARUCO_DICT.get(args["type"], None) is None:
#         print(f"ArUCo tag type '{args['type']}' is not supported")
#         sys.exit(0)

#     aruco_dict_type = ARUCO_DICT[args["type"]]
#     calibration_matrix_path = args["K_Matrix"]
#     distortion_coefficients_path = args["D_Coeff"]
    
#     k = np.load(calibration_matrix_path)
#     d = np.load(distortion_coefficients_path)

#     video = cv2.VideoCapture(0)
#     time.sleep(2.0)

#     while True:
#         ret, frame = video.read()

#         if not ret:
#             break
        
#         output = pose_esitmation(frame, aruco_dict_type, k, d)

#         cv2.imshow('Estimated Pose', output)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()






# import numpy as np
# import cv2
# import sys
# from utils import ARUCO_DICT
# import argparse
# import time

# def draw_axis(frame, corners, rvec, tvec, matrix_coefficients, distortion_coefficients):
#     """
#     Draws the axes on the detected marker.
#     frame: Input frame
#     corners: Detected corners of the marker
#     rvec: Rotation vector
#     tvec: Translation vector
#     matrix_coefficients: Intrinsic camera matrix
#     distortion_coefficients: Camera distortion coefficients
#     """
#     # Axis length for drawing (3D to 2D)
#     axis_length = 0.02

#     # Define axis points in 3D
#     axis_points = np.float32([
#         [axis_length, 0, 0], 
#         [0, axis_length, 0], 
#         [0, 0, axis_length],
#         [0, 0, 0]
#     ]).reshape(-1, 3)

#     # Project 3D points to 2D image plane
#     img_pts, jacobian = cv2.projectPoints(axis_points, rvec, tvec, matrix_coefficients, distortion_coefficients)

#     # Convert the points to integer format
#     img_pts = np.int32(img_pts).reshape(-1, 2)

#     # Draw the axes on the frame
#     frame = cv2.line(frame, tuple(img_pts[3]), tuple(img_pts[0]), (255, 0, 0), 5) # X axis - Red
#     frame = cv2.line(frame, tuple(img_pts[3]), tuple(img_pts[1]), (0, 255, 0), 5) # Y axis - Green
#     frame = cv2.line(frame, tuple(img_pts[3]), tuple(img_pts[2]), (0, 0, 255), 5) # Z axis - Blue

#     return frame

# def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
#     '''
#     frame - Frame from the video stream
#     matrix_coefficients - Intrinsic matrix of the calibrated camera
#     distortion_coefficients - Distortion coefficients associated with your camera

#     return:-
#     frame - The frame with the axis drawn on it
#     '''

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters()

#     # Initialize the detector
#     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#     # Detect markers
#     corners, ids, rejected_img_points = detector.detectMarkers(gray)

#     # If markers are detected
#     if len(corners) > 0:
#         for i in range(0, len(ids)):
#             # Estimate pose of each marker and return the values rvec and tvec
#             rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                            distortion_coefficients)
#             # Draw a square around the markers
#             cv2.aruco.drawDetectedMarkers(frame, corners)

#             # Draw the axes
#             frame = draw_axis(frame, corners[i], rvec, tvec, matrix_coefficients, distortion_coefficients)

#     return frame

# if __name__ == '__main__':
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
#     ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
#     ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
#     args = vars(ap.parse_args())

#     # Check if the specified ArUco dictionary type is supported
#     if ARUCO_DICT.get(args["type"], None) is None:
#         print(f"ArUCo tag type '{args['type']}' is not supported")
#         sys.exit(0)

#     aruco_dict_type = ARUCO_DICT[args["type"]]
#     calibration_matrix_path = args["K_Matrix"]
#     distortion_coefficients_path = args["D_Coeff"]

#     # Load the camera matrix and distortion coefficients from files
#     k = np.load(calibration_matrix_path)
#     d = np.load(distortion_coefficients_path)

#     # Start video capture
#     video = cv2.VideoCapture(0)
#     time.sleep(2.0)

#     while True:
#         ret, frame = video.read()

#         if not ret:
#             break

#         output = pose_estimation(frame, aruco_dict_type, k, d)

#         # Display the resulting frame
#         cv2.imshow('Estimated Pose', output)

#         # Press 'q' to quit
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()



import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

def draw_axes(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, length=0.01):
    # Define the 3D coordinates of the axis end points
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)

    # Project 3D points to the image plane
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, matrix_coefficients, distortion_coefficients)
    
    # Convert points to integer tuples for drawing
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Ensure we have the correct number of points
    if len(imgpts) < 3:
        print("Not enough points to draw axes.")
        return frame
    
    origin = tuple(imgpts[0].ravel())
    x_axis = tuple(imgpts[1].ravel())
    y_axis = tuple(imgpts[2].ravel())

    # Draw the axis lines
    cv2.line(frame, origin, x_axis, (0, 0, 255), 3)  # Red for X axis
    cv2.line(frame, origin, y_axis, (0, 255, 0), 3)  # Green for Y axis
    
    return frame


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    if len(corners) > 0:
        for i in range(len(ids)):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.02, matrix_coefficients, distortion_coefficients
            )

            # Draw detected markers and their axis
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for j in range(len(ids)):
                # Draw Axis
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec[j], tvec[j], 0.02)

                # Calculate the rotation matrix and extract Euler angles
                R, _ = cv2.Rodrigues(rvec[j])
                pitch, roll, yaw = rotationMatrixToEulerAngles(R)

                # Display the pitch, roll, and yaw angles
                cv2.putText(frame, f"X: {pitch:.2f}", (10, 30 + j * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"Y: {roll:.2f}", (10, 60 + j * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {yaw:.2f}", (10, 90 + j * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def rotationMatrixToEulerAngles(R):
    """
    Calculate the pitch, roll, and yaw from the rotation matrix.
    """
    assert(R.shape == (3, 3))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
