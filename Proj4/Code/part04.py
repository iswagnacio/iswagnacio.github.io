import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path


def load_calibration(calibration_file='camera_calibration.npz'):
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs


def load_poses(pose_file='camera_poses.npz'):
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    data = np.load(pose_file)
    c2w_matrices = data['c2w_matrices']
    return c2w_matrices


def get_successful_images(image_folder, num_poses):
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + 
                        glob.glob(os.path.join(image_folder, '*.png')) +
                        glob.glob(os.path.join(image_folder, '*.jpeg')) +
                        glob.glob(os.path.join(image_folder, '*.JPG')) +
                        glob.glob(os.path.join(image_folder, '*.JPEG')))
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    successful_images = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        if ids is not None and len(ids) > 0:
            successful_images.append(image_path)

    if len(successful_images) != num_poses:
        print(f"Found {len(successful_images)} successful images "
              f"but have {num_poses} poses.")
        print(f"Using first {min(len(successful_images), num_poses)} matches")
        successful_images = successful_images[:num_poses]
    
    return successful_images


def test_and_handle_undistortion(test_image, camera_matrix, dist_coeffs):
    h, w = test_image.shape[:2]
    undistorted_std = cv2.undistort(test_image, camera_matrix, dist_coeffs)
    original_black = np.sum(np.all(test_image == 0, axis=2))
    undistorted_black = np.sum(np.all(undistorted_std == 0, axis=2))
    has_black_boundaries = undistorted_black > original_black * 3
    
    if has_black_boundaries:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h)
        )
        undistorted_opt = cv2.undistort(test_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        x, y, w_roi, h_roi = roi
        
        if w_roi > 0 and h_roi > 0:
            cropped = undistorted_opt[y:y+h_roi, x:x+w_roi]
            adjusted_camera_matrix = new_camera_matrix.copy()
            adjusted_camera_matrix[0, 2] -= x  # cx
            adjusted_camera_matrix[1, 2] -= y  # cy
            return 'crop', adjusted_camera_matrix, (x, y, w_roi, h_roi), cropped
        
        else:
            return 'standard', camera_matrix, None, undistorted_std
    else:
        return 'standard', camera_matrix, None, undistorted_std


def process_all_images(image_paths, camera_matrix, dist_coeffs, target_size=(400, 300)):

    if len(image_paths) == 0:
        return np.array([]), camera_matrix
 
    test_image = cv2.imread(image_paths[0])
    strategy, final_camera_matrix, roi_info, demo_result = test_and_handle_undistortion(
        test_image, camera_matrix, dist_coeffs
    )

    processed_images = []
    
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        if strategy == 'crop':
            x, y, w_roi, h_roi = roi_info
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, final_camera_matrix)
            processed = undistorted[y:y+h_roi, x:x+w_roi]
        else:
            processed = cv2.undistort(img, camera_matrix, dist_coeffs)
        
        processed_images.append(processed)
    
    processed_images = np.array(processed_images)
    current_h, current_w = processed_images[0].shape[:2]

    target_w, target_h = target_size
    if (current_w, current_h) != (target_w, target_h):
        scale_x = target_w / current_w
        scale_y = target_h / current_h
        resized_images = []
        for img in processed_images:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized_images.append(resized)
        processed_images = np.array(resized_images)

        final_camera_matrix[0, 0] *= scale_x  # fx
        final_camera_matrix[1, 1] *= scale_y  # fy  
        final_camera_matrix[0, 2] *= scale_x  # cx
        final_camera_matrix[1, 2] *= scale_y  # cy

    rgb_images = []
    for img in processed_images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32)
        rgb_images.append(img_normalized)
    
    final_images = np.array(rgb_images, dtype=np.float32)
    return final_images, final_camera_matrix


def convert_poses_to_4x4(c2w_3x4):
    n = len(c2w_3x4)
    c2w_4x4 = np.zeros((n, 4, 4), dtype=np.float32)
    c2w_4x4[:, :3, :] = c2w_3x4
    c2w_4x4[:, 3, 3] = 1.0
    return c2w_4x4


def split_data(images, c2ws):
    n = len(images)
    indices = np.arange(n)
    np.random.seed(42)

    train_images = images
    train_poses = c2ws

    n_val = max(1, int(n * 0.1))
    val_indices = np.random.choice(n, n_val, replace=False)
    val_images = images[val_indices]
    val_poses = c2ws[val_indices]
    base_pose = c2ws[n//2].copy()
    test_poses = []
    
    n_test_views = 6
    for i in range(n_test_views):
        angle = (i / n_test_views) * 2 * np.pi
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotation_z = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a,  cos_a, 0, 0],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ], dtype=np.float32)

        rotated_pose = rotation_z @ base_pose
        test_poses.append(rotated_pose)

    test_poses = np.array(test_poses)

    return {
        'images_train': train_images,
        'c2ws_train': train_poses,
        'images_val': val_images, 
        'c2ws_val': val_poses,
        'c2ws_test': test_poses
    }


def create_nerf_dataset(image_folder, calibration_file, pose_file, 
                       output_file='nerf_data.npz', target_size=(214, 285)):
    
    camera_matrix, dist_coeffs = load_calibration(calibration_file)
    c2w_matrices = load_poses(pose_file)
    image_paths = get_successful_images(image_folder, len(c2w_matrices))
    processed_images, final_camera_matrix = process_all_images(
        image_paths, camera_matrix, dist_coeffs, target_size
    )

    c2w_4x4 = convert_poses_to_4x4(c2w_matrices)
    data_split = split_data(processed_images, c2w_4x4)
    focal_x = final_camera_matrix[0, 0]
    focal_y = final_camera_matrix[1, 1]
    focal = (focal_x + focal_y) / 2.0

    np.savez_compressed(
        output_file,
        **data_split,
        focal=focal
    )
    
    return output_file


def main():
    calibration_file = 'Proj4/Code/calibration.npz'
    pose_file = 'Proj4/Code/poses.npz'
    image_folder = './Proj4/Media/Object'
    output_file = 'nerf.npz'
    target_size = (400, 300)
    
    try:
        create_nerf_dataset(
            image_folder=image_folder,
            calibration_file=calibration_file, 
            pose_file=pose_file,
            output_file=output_file,
            target_size=target_size
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()