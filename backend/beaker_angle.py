# import cv2
# import numpy as np

# # Load the beaker image
# beaker_image = cv2.imread('beaker-alpha.png', cv2.IMREAD_UNCHANGED)

# # Load camera calibration parameters
# calibration_matrix_path = 'calibration_matrix.npy'
# distortion_coefficients_path = 'distortion_coefficients.npy'
# matrix_coefficients = np.load(calibration_matrix_path)
# distortion_coefficients = np.load(distortion_coefficients_path)

# def rotation_matrix_to_euler_angles(R):
#     sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
#     singular = sy < 1e-6
    
#     if not singular:
#         x = np.arctan2(R[2, 1], R[2, 2])
#         y = np.arctan2(-R[2, 0], sy)
#         z = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         x = np.arctan2(-R[1, 2], R[1, 1])
#         y = np.arctan2(-R[2, 0], sy)
#         z = 0
    
#     return np.degrees(x), np.degrees(y), np.degrees(z)  # Convert to degrees

# def overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image):
#     if ids is not None:
#         for i, marker_id in enumerate(ids.flatten()):
#             if marker_id == 0:
#                 marker_corners = corners[i].reshape((4, 2))

#                 # Get the four corners of the marker
#                 top_left, top_right, bottom_right, bottom_left = marker_corners

#                 marker_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

#                 # Define the points for the beaker image to be warped
#                 beaker_points = np.array([[0, 0], [beaker_image.shape[1] - 1, 0], 
#                                           [beaker_image.shape[1] - 1, beaker_image.shape[0] - 1], 
#                                           [0, beaker_image.shape[0] - 1]], dtype=np.float32)

#                 # Compute the perspective transform matrix
#                 M = cv2.getPerspectiveTransform(beaker_points, marker_points)
#                 warped_beaker = cv2.warpPerspective(beaker_image, M, (frame.shape[1], frame.shape[0]), 
#                                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

#                 # Overlay the warped beaker image on the frame
#                 mask = warped_beaker[:, :, 3]
#                 mask = cv2.merge([mask, mask, mask])
#                 beaker_rgb = warped_beaker[:, :, :3]
#                 frame = np.where(mask == 0, frame, beaker_rgb)
                
#                 # Get the rotation matrix and calculate angles
#                 rvec = rvecs[i]
#                 tvec = tvecs[i]
#                 rotation_matrix, _ = cv2.Rodrigues(rvec)
#                 pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
                
#                 # Print the angles on the frame
#                 text = f"X-Inclination: {pitch:.2f}, Y-Inclination: {roll:.2f}, Z-Inclination: {yaw:.2f}"
#                 cv2.putText(frame, text, (int(top_left[0]), int(top_left[1]) - 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame

# def main():
#     cap = cv2.VideoCapture(0)

#     # Load ArUco dictionary and parameters
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
#     parameters = cv2.aruco.DetectorParameters()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

#         if ids is not None:
#             # Estimate pose of the detected markers
#             rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)

#             # Overlay the beaker and display angles
#             output_frame = overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image)
#         else:
#             output_frame = frame

#         cv2.imshow('Augmented Reality', output_frame)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()





# import cv2
# import numpy as np

# # Load the beaker image
# beaker_image = cv2.imread('beaker-alpha1.png', cv2.IMREAD_UNCHANGED)

# # Load camera calibration parameters
# calibration_matrix_path = 'calibration_matrix.npy'
# distortion_coefficients_path = 'distortion_coefficients.npy'
# matrix_coefficients = np.load(calibration_matrix_path)
# distortion_coefficients = np.load(distortion_coefficients_path)

# def overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image):
#     if ids is not None:
#         for i, marker_id in enumerate(ids.flatten()):
#             if marker_id == 0:
#                 marker_corners = corners[i].reshape((4, 2))

#                 # Get the four corners of the marker
#                 top_left, top_right, bottom_right, bottom_left = marker_corners

#                 marker_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

#                 # Define the points for the beaker image to be warped
#                 beaker_points = np.array([[0, 0], [beaker_image.shape[1] - 1, 0], 
#                                           [beaker_image.shape[1] - 1, beaker_image.shape[0] - 1], 
#                                           [0, beaker_image.shape[0] - 1]], dtype=np.float32)

#                 # Compute the perspective transform matrix
#                 M = cv2.getPerspectiveTransform(beaker_points, marker_points)
#                 warped_beaker = cv2.warpPerspective(beaker_image, M, (frame.shape[1], frame.shape[0]), 
#                                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

#                 # Split the warped beaker image into RGB and alpha channels
#                 if warped_beaker.shape[2] == 4:
#                     beaker_rgb = warped_beaker[:, :, :3]
#                     beaker_alpha = warped_beaker[:, :, 3] / 255.0
#                 else:
#                     beaker_rgb = warped_beaker
#                     beaker_alpha = np.ones((warped_beaker.shape[0], warped_beaker.shape[1]))

#                 # Create the mask for overlay
#                 mask = (beaker_alpha > 0).astype(np.uint8) * 255
#                 mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
#                 # Blend the beaker image with the frame
#                 frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask[:, :, 0]))
#                 frame += cv2.bitwise_and(beaker_rgb, beaker_rgb, mask=mask[:, :, 0])

#                 # Get the rotation matrix and calculate angles
#                 rvec = rvecs[i]
#                 tvec = tvecs[i]
#                 rotation_matrix, _ = cv2.Rodrigues(rvec)
#                 pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
                
#                 # Print the angles on the frame
#                 text = f"X: {pitch:.2f}, Y: {roll:.2f}, Z: {yaw:.2f}"
#                 cv2.putText(frame, text, (int(top_left[0]), int(top_left[1]) - 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame

# def rotation_matrix_to_euler_angles(R):
#     sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
#     singular = sy < 1e-6
    
#     if not singular:
#         x = np.arctan2(R[2, 1], R[2, 2])
#         y = np.arctan2(-R[2, 0], sy)
#         z = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         x = np.arctan2(-R[1, 2], R[1, 1])
#         y = np.arctan2(-R[2, 0], sy)
#         z = 0
    
#     return np.degrees(x), np.degrees(y), np.degrees(z)  # Convert to degrees

# def main():
#     cap = cv2.VideoCapture(0)

#     # Load ArUco dictionary and parameters
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
#     parameters = cv2.aruco.DetectorParameters()  # Correct method for DetectorParameters

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

#         if ids is not None:
#             # Estimate pose of the detected markers
#             rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)

#             # Overlay the beaker and display angles
#             output_frame = overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image)
#         else:
#             output_frame = frame

#         cv2.imshow('Augmented Reality', output_frame)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()






import cv2
import numpy as np

# Load the beaker image with alpha channel
beaker_image = cv2.imread('beaker-alpha1.png', cv2.IMREAD_UNCHANGED)

# Load camera calibration parameters
calibration_matrix_path = 'calibration_matrix.npy'
distortion_coefficients_path = 'distortion_coefficients.npy'
matrix_coefficients = np.load(calibration_matrix_path)
distortion_coefficients = np.load(distortion_coefficients_path)

# def overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image):
#     if ids is not None:
#         for i, marker_id in enumerate(ids.flatten()):
#             if marker_id == 0:
#                 marker_corners = corners[i].reshape((4, 2))

#                 # # Get the four corners of the marker
#                 # top_left, top_right, bottom_right, bottom_left = marker_corners

#                 # marker_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

#                 # # Define the points for the beaker image to be warped
#                 # beaker_points = np.array([[0, 0], [beaker_image.shape[1] - 1, 0], 
#                 #                           [beaker_image.shape[1] - 1, beaker_image.shape[0] - 1], 
#                 #                           [0, beaker_image.shape[0] - 1]], dtype=np.float32)

#                 # Get the four corners of the marker
#                 top_left, top_right, bottom_right, bottom_left = marker_corners

#                 # Ensure the marker_points are ordered correctly in a clockwise manner
#                 marker_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

#                 # Ensure the beaker image has the same aspect ratio as the marker points
#                 beaker_points = np.array([[0, 0], [beaker_image.shape[1] - 1, 0], 
#                                         [beaker_image.shape[1] - 1, beaker_image.shape[0] - 1], 
#                                         [0, beaker_image.shape[0] - 1]], dtype=np.float32)

#                 # Compute the perspective transform matrix 'M'
#                 M = cv2.getPerspectiveTransform(beaker_points, marker_points)


#                 # Compute the perspective transform matrix
#                 warped_beaker = cv2.warpPerspective(beaker_image, M, (frame.shape[1], frame.shape[0]), 
#                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


#                 # Split the warped beaker image into RGB and alpha channels
#                 if warped_beaker.shape[2] == 4:  # Ensure the image has 4 channels (RGBA)
#                     beaker_rgb = warped_beaker[:, :, :3]
#                     beaker_alpha = warped_beaker[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
#                 else:
#                     beaker_rgb = warped_beaker
#                     beaker_alpha = np.ones((warped_beaker.shape[0], warped_beaker.shape[1]))

#                 # Blend the warped beaker image with the frame
#                 for c in range(0, 3):  # Iterate over the color channels
#                     frame[:, :, c] = frame[:, :, c] * (1 - beaker_alpha) + beaker_rgb[:, :, c] * beaker_alpha


#                 # Get the rotation matrix and calculate angles
#                 rvec = rvecs[i]
#                 tvec = tvecs[i]
#                 rotation_matrix, _ = cv2.Rodrigues(rvec)
#                 pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
                
#                 # Print the angles on the frame
#                 text = f"X: {pitch:.2f}, Y: {roll:.2f}, Z: {yaw:.2f}"
#                 cv2.putText(frame, text, (int(top_left[0]), int(top_left[1]) - 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame

def overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image):
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 0:
                marker_corners = corners[i].reshape((4, 2))

                # Get the four corners of the marker
                top_left, top_right, bottom_right, bottom_left = marker_corners

                marker_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

                # Define the points for the beaker image to be warped
                beaker_points = np.array([[0, 0], [beaker_image.shape[1] - 1, 0], 
                                          [beaker_image.shape[1] - 1, beaker_image.shape[0] - 1], 
                                          [0, beaker_image.shape[0] - 1]], dtype=np.float32)

                # Compute the perspective transform matrix
                M = cv2.getPerspectiveTransform(beaker_points, marker_points)
                warped_beaker = cv2.warpPerspective(beaker_image, M, (frame.shape[1], frame.shape[0]), 
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                # Split the warped beaker image into RGB and alpha channels
                if warped_beaker.shape[2] == 4:
                    beaker_rgb = warped_beaker[:, :, :3]
                    beaker_alpha = warped_beaker[:, :, 3] / 255.0
                else:
                    beaker_rgb = warped_beaker
                    beaker_alpha = np.ones((warped_beaker.shape[0], warped_beaker.shape[1]))

                # Blend the beaker image with the frame
                for c in range(0, 3):  # Iterate over the color channels
                    frame[:, :, c] = frame[:, :, c] * (1 - beaker_alpha) + beaker_rgb[:, :, c] * beaker_alpha

                # Get the rotation matrix and calculate angles
                rvec = rvecs[i]
                tvec = tvecs[i]
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

                # Print the angles on the frame
                text = f"X: {pitch:.2f}, Y: {roll:.2f}, Z: {yaw:.2f}"
                cv2.putText(frame, text, (int(top_left[0]), int(top_left[1]) - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show "Chemical Pouring" only when Y (roll) is greater than 3 degrees
                if roll > 3:  # Checking only the roll (Y-axis rotation)
                    cv2.putText(frame, 'Chemical Pouring', (int(top_left[0]), int(top_left[1]) - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame



def rotation_matrix_to_euler_angles(R):
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
    
    return np.degrees(x), np.degrees(y), np.degrees(z)  # Convert to degrees

def main():
    cap = cv2.VideoCapture(0)

    # Load ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()  # Correct method for DetectorParameters

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None:
            # Estimate pose of the detected markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)

            # Overlay the beaker and display angles
            output_frame = overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image)
        else:
            output_frame = frame

        cv2.imshow('Augmented Reality', output_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
