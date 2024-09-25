
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import base64
from io import BytesIO
import time

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "http://localhost:3000"}})

# Load beaker image and camera calibration matrices
beaker_image = cv2.imread('beaker-alpha1.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load camera calibration parameters
calibration_matrix_path = 'calibration_matrix.npy'
distortion_coefficients_path = 'distortion_coefficients.npy'
matrix_coefficients = np.load(calibration_matrix_path)
distortion_coefficients = np.load(distortion_coefficients_path)

# Define the step sequence and corresponding marker IDs
step_sequence = {
    0: 0,  # Marker ID 0 corresponds to step 1 (e.g., showing distilled water)
    1: 1,  # Marker ID 1 corresponds to step 2 (e.g., showing test tube 1)
    2: 2,  # Marker ID 2 corresponds to step 3 (e.g., adding HCl)
    3: 3,  # Marker ID 3 corresponds to step 4 (e
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12  # Marker ID 12 corresponds to the final step (completion)
}

# Mapping marker IDs to names
marker_names = {
    0: "Distilled Water",
    1: "Test Tube 1",
    2: "HCL",
    3: "Barium Nitrate",
    4: "Sulphuric Acid"
}

current_step = 0  # Initialize the current step to 1

# Global variable to keep track of consecutive detection frames
consecutive_detection_frames = 0
detection_threshold = 5  # Number of frames required to increment step

# Track the alternating y-axis state for step 7
alternate_yaxis_condition_met = [False, False]  # First element for >3, second for <3



def overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image):
    global current_step, consecutive_detection_frames, alternate_yaxis_condition_met
    detected = False
    required_markers_detected = {'id_1': False, 'id_3': False, 'id_0': False, 'id_2': False}  # For previous steps
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            _, roll, _ = rotation_matrix_to_euler_angles(rotation_matrix)
            
            corner = corners[i].reshape((4, 2))
            top_left = tuple(corner[0].astype(int))
            # Get the name of the marker based on marker_id
            marker_name = marker_names.get(marker_id, f"Unknown Marker ID: {marker_id}")
            cv2.putText(frame, f'ID: {marker_name}, Roll: {roll:.2f}', top_left, cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Step sequence 5: Check for markers with ID=1 and ID=3
            if current_step == 5:
                if marker_id == 1 and roll < 3:
                    required_markers_detected['id_1'] = True
                    cv2.putText(frame, 'Marker 1 Detected with y-axis < 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 0, 0), 2, cv2.LINE_AA)
                if marker_id == 3 and roll > 3:
                    required_markers_detected['id_3'] = True
                    cv2.putText(frame, 'Marker 3 Detected with y-axis > 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 0, 0), 2, cv2.LINE_AA)

                if required_markers_detected['id_1'] and required_markers_detected['id_3']:
                    detected = True
                    consecutive_detection_frames += 1


                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1
                        consecutive_detection_frames = 0
                        break

            # Step sequence 6: Check for markers with ID=0 and ID=1
            elif current_step == 6:
                if marker_id == 0 and roll > 3:
                    required_markers_detected['id_0'] = True
                    cv2.putText(frame, 'Marker 0 Detected with y-axis > 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)
                if marker_id == 1 and roll < 3:
                    required_markers_detected['id_1'] = True
                    cv2.putText(frame, 'Marker 1 Detected with y-axis < 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)

                if required_markers_detected['id_0'] and required_markers_detected['id_1']:
                    detected = True
                    consecutive_detection_frames += 1
                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1  # Move to next step
                        consecutive_detection_frames = 0
                        break

            # Step sequence 7: Detect marker ID=1 with alternating roll conditions
            elif current_step == 7:
                if marker_id == 1:
                    if roll > 3 and not alternate_yaxis_condition_met[0]:  # First condition: roll > 3
                        alternate_yaxis_condition_met[1] = True
                        cv2.putText(frame, 'Marker 1 Detected with y-axis > 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 255, 0), 2, cv2.LINE_AA)

                    elif roll < 3 and alternate_yaxis_condition_met[0] and not alternate_yaxis_condition_met[1]:  # Second condition: roll < 3
                        alternate_yaxis_condition_met[1] = True
                        cv2.putText(frame, 'Marker 1 Detected with y-axis < 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 255, 0), 2, cv2.LINE_AA)

                    elif roll > 3 and alternate_yaxis_condition_met[1]:  # Third condition: roll > 3 again
                        cv2.putText(frame, 'Marker 1 Detected with y-axis > 3 again', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 255, 0), 2, cv2.LINE_AA)
                        consecutive_detection_frames += 1
                        if consecutive_detection_frames >= detection_threshold:
                            detected = True
                            # Reset conditions and move to the next step
                            current_step += 1
                            consecutive_detection_frames = 0
                            alternate_yaxis_condition_met = [False, False]
                            break
            # Step sequence 8: Check for markers with ID=1 and ID=2
            elif current_step == 8:
                if marker_id == 1 and roll > 3:
                    required_markers_detected['id_1'] = True
                    cv2.putText(frame, 'Marker 1 Detected with y-axis > 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)
                if marker_id == 2 and roll < 3:
                    required_markers_detected['id_2'] = True
                    cv2.putText(frame, 'Marker 2 Detected with y-axis < 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)

                if required_markers_detected['id_1'] and required_markers_detected['id_2']:
                    detected = True
                    consecutive_detection_frames += 1
                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1  # Move to next step
                        consecutive_detection_frames = 0
                        break
            # Step sequence 9: Check for markers with ID=1 and ID=2
            elif current_step == 9:
                if marker_id == 1 and roll > 3:
                    required_markers_detected['id_1'] = True
                    cv2.putText(frame, 'Marker 1 Detected with y-axis > 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)
                if marker_id == 3 and roll < 3:
                    required_markers_detected['id_3'] = True
                    cv2.putText(frame, 'Marker 3 Detected with y-axis < 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)

                if required_markers_detected['id_1'] and required_markers_detected['id_2']:
                    detected = True
                    consecutive_detection_frames += 1
                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1  # Move to next step
                        consecutive_detection_frames = 0
                        break
            # Step sequence 10: Check for markers with ID=1 and ID=2
            elif current_step == 10:
                if marker_id == 1 and roll < 3:
                    required_markers_detected['id_1'] = True
                    cv2.putText(frame, 'Marker 1 Detected with y-axis > 3', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)
                if marker_id == 4 and roll > 3:
                    required_markers_detected['id_3'] = True
                    cv2.putText(frame, 'Marker 4 Detected with y-axis < 3', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 255), 2, cv2.LINE_AA)

                if required_markers_detected['id_1'] and required_markers_detected['id_2']:
                    detected = True
                    consecutive_detection_frames += 1
                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1  # Move to next step
                        consecutive_detection_frames = 0
                        break

            else:
                # Handle other steps with single marker detection
                expected_marker_id = step_sequence.get(current_step, -1)
                if marker_id == expected_marker_id:
                    consecutive_detection_frames += 1
                    if consecutive_detection_frames >= detection_threshold:
                        current_step += 1
                        consecutive_detection_frames = 0
                        detected = True
                else:
                    consecutive_detection_frames = 0

            # Debug: Show the current step on the frame
            cv2.putText(frame, f'Step: {current_step}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, detected






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

# @app.route('/detect', methods=['POST'])
# def detect_aruco():
#     # Get the image from the request
#     file = request.files['image']
#     npimg = np.fromfile(file, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Load ArUco dictionary and detect markers
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
#     parameters = cv2.aruco.DetectorParameters()  
#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

#     if ids is not None:
#         print(f"Detected marker IDs: {ids.flatten()}")  # Print detected marker IDs
#         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)
#         output_frame, detected = overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image)

#         # Convert ids to a list
#         ids_list = ids.flatten().tolist() if ids is not None else []
#     else:
#         print("No markers detected.") 
#         output_frame = frame
#         detected = False
#         ids_list = []

#     # Encode the image to base64
#     _, buffer = cv2.imencode('.jpg', output_frame)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')
#     # print('img----------------  ', img_base64)
#     # cv2.imshow('img', img_base64)

#     response = {
#         'detected': detected,
#         'augmented_image': img_base64,
#         'ids': ids_list,  # Add ids to the response
#         'current_step': current_step  # Send the current step to the frontend
#     }
#     print(f"Sending response: detected={detected}, ids={ids_list}, augmented_image_length={len(img_base64) if detected else 0}")  # Debug statement
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)



@app.route('/detect', methods=['POST'])
def detect_aruco():
    # Get the image from the request
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary and detect markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()  
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    detected_rolls = {}  # Dictionary to store roll values for detected markers

    if ids is not None:
        print(f"Detected marker IDs: {ids.flatten()}")  # Print detected marker IDs
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)
        
        # Collect roll values for detected markers
        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            _, roll, _ = rotation_matrix_to_euler_angles(rotation_matrix)
            detected_rolls[int(marker_id)] = roll  # Convert marker_id to int

        output_frame, detected = overlay_beaker_on_marker(frame, corners, ids, rvecs, tvecs, beaker_image)

        # Convert ids to a list
        ids_list = ids.flatten().tolist() if ids is not None else []
    else:
        print("No markers detected.") 
        output_frame = frame
        detected = False
        ids_list = []

    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', output_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    response = {
        'detected': detected,
        'augmented_image': img_base64,
        'ids': ids_list,
        'roll': detected_rolls,  # Include roll values in the response
        'currentStep': current_step  # Send the current step to the frontend
    }
    
    print(f"Sending response: detected={detected}, ids={ids_list}, rolls={detected_rolls}, augmented_image_length={len(img_base64) if detected else 0}")  # Debug statement
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
