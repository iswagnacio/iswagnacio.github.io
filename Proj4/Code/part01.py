import cv2
import numpy as np
import glob
import os
from pathlib import Path


def detect_aruco_tags(image, aruco_dict, aruco_params):
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(image)
    return corners, ids

def prepare_calibration_data(image_folder, tag_size=0.055, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()

    tag_positions = {
        0: (0.0, 0.0),          # Top-left
        1: (0.090, 0.0),        # Top-right  
        2: (0.0, 0.07567),      # Middle-left
        3: (0.090, 0.07567),    # Middle-right
        4: (0.0, 0.15134),      # Bottom-left
        5: (0.090, 0.15134)     # Bottom-right
    }
    
    object_points_list = []
    image_points_list = []
    image_size = None

    image_files = sorted(
        glob.glob(os.path.join(image_folder, '*.jpeg')) +
        glob.glob(os.path.join(image_folder, '*.jpg')) +
        glob.glob(os.path.join(image_folder, '*.JPG'))
    )
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    successful_detections = 0
    total_tag_detections = 0
    
    for idx, image_path in enumerate(image_files):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0]) 

        corners, ids = detect_aruco_tags(gray, aruco_dict, aruco_params)

        if ids is not None and len(ids) > 0:
            image_tag_count = 0
            for i in range(len(ids)):
                tag_id = ids[i][0]
                
                if tag_id not in tag_positions:
                    print(f"Unknown tag ID {tag_id}, skipping")
                    continue

                tag_x, tag_y = tag_positions[tag_id]
                object_points = np.array([
                    [tag_x, tag_y, 0],
                    [tag_x + tag_size, tag_y, 0],
                    [tag_x + tag_size, tag_y + tag_size, 0],
                    [tag_x, tag_y + tag_size, 0]
                ], dtype=np.float32)
                
                marker_corners = corners[i].reshape(4, 2)
                object_points_list.append(object_points)
                image_points_list.append(marker_corners.astype(np.float32))
                image_tag_count += 1
            
            if image_tag_count > 0:
                successful_detections += 1
                total_tag_detections += image_tag_count
                print(f"Image {idx+1}/{len(image_files)}: Detected {image_tag_count} valid tag(s) with IDs: {ids.flatten()}")
            else:
                print(f"Image {idx+1}/{len(image_files)}: No valid tags detected - skipping")
        else:
            print(f"Image {idx+1}/{len(image_files)}: No tags detected - skipping")
    
    if len(object_points_list) == 0:
        raise ValueError("No ArUco tags detected")
    
    print(f"Total: {successful_detections} successful images, {total_tag_detections} tag detections")
    return object_points_list, image_points_list, image_size

def calibrate_camera(object_points_list, image_points_list, image_size):

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None
    )

    print(f"  RMS Reprojection Error: {ret:.4f} pixels")
    return camera_matrix, dist_coeffs, rvecs, tvecs, ret

def save_calibration_results(camera_matrix, dist_coeffs, output_file='camera_calibration.npz'):
    np.savez(output_file,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print(f"\nCalibration saved")

def main():

    calibration_image_folder = 'Proj4/Media/Calibration'
    tag_size = 0.055
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    output_file = 'calibration.npz'
    
    try:
        object_points_list, image_points_list, image_size = prepare_calibration_data(
            calibration_image_folder, 
            tag_size=tag_size,
            aruco_dict_type=aruco_dict_type
        )
        camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = calibrate_camera(
            object_points_list,
            image_points_list,
            image_size
        )
        save_calibration_results(camera_matrix, dist_coeffs, output_file)
          
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()