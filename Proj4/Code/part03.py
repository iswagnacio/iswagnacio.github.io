import cv2
import numpy as np
import glob
import os
import time
import viser
scale = 10.0

def load_calibration(calibration_file='camera_calibration.npz'):
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    return camera_matrix, dist_coeffs


def detect_aruco_tag(image, aruco_dict, aruco_params):
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    
    corners, ids, rejected = cv2.aruco.detectMarkers(
        image, aruco_dict, parameters=aruco_params
    )
    return corners, ids


def estimate_pose_no_filter(image_path, camera_matrix, dist_coeffs, tag_size, 
                            aruco_dict, aruco_params):
    image = cv2.imread(image_path)
    if image is None:
        return False, None, None, None, float('inf')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    corners, ids = detect_aruco_tag(gray, aruco_dict, aruco_params)
    
    if ids is None or len(ids) == 0:
        print(f"No ArUco tags detected in {os.path.basename(image_path)}")
        return False, None, image, image_rgb, float('inf')
    
    print(f"Detected {len(ids)} tags with IDs: {ids.flatten()}")
    
    tag_positions = {
        0: (0.0, 0.0),          # Top-left
        1: (0.090, 0.0),        # Top-right  
        2: (0.0, 0.07567),      # Middle-left
        3: (0.090, 0.07567),    # Middle-right
        4: (0.0, 0.15134),      # Bottom-left
        5: (0.090, 0.15134)     # Bottom-right
    }

    all_object_points = []
    all_image_points = []
    
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
        
        image_points = corners[i].reshape(4, 2).astype(np.float32)
        
        all_object_points.extend(object_points)
        all_image_points.extend(image_points)
    
    if len(all_object_points) == 0:
        print(f"No valid tags found")
        return False, None, image, image_rgb, float('inf')

    all_object_points = np.array(all_object_points, dtype=np.float32)
    all_image_points = np.array(all_image_points, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(
        all_object_points,
        all_image_points,
        camera_matrix,
        dist_coeffs
    )
    
    if not success:
        print(f"solvePnP failed for {os.path.basename(image_path)}")
        return False, None, image, image_rgb, float('inf')

    projected_points, _ = cv2.projectPoints(
        all_object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojection_error = np.mean(np.linalg.norm(
        projected_points.reshape(-1, 2) - all_image_points, axis=1
    ))
    
    print(f"Reprojection error: {reprojection_error:.2f} pixels")
    
    R, _ = cv2.Rodrigues(rvec)
    
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    c2w_full = np.linalg.inv(w2c)
    grid_center = np.array([0.045, 0.07567, 0.0])

    T_center_transform = np.eye(4)
    T_center_transform[:3, 3] = -grid_center
    c2w_centered = c2w_full @ T_center_transform

    scale_factor = scale
    c2w_centered[:3, 3] *= scale_factor
    c2w = c2w_centered[:3, :]
    
    return True, c2w, image, image_rgb, reprojection_error


def process_object_images(image_folder, camera_matrix, dist_coeffs, 
                         tag_size=0.06, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + 
                        glob.glob(os.path.join(image_folder, '*.png')) +
                        glob.glob(os.path.join(image_folder, '*.jpeg')) +
                        glob.glob(os.path.join(image_folder, '*.JPG')) +
                        glob.glob(os.path.join(image_folder, '*.JPEG')))
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    pose_candidates = []
    for idx, image_path in enumerate(image_files):
        print(f"\nProcessing image {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        success, c2w, image, image_rgb, reprojection_error = estimate_pose_no_filter(
            image_path, camera_matrix, dist_coeffs, tag_size,
            aruco_dict, aruco_params
        )
        
        if success:
            pose_candidates.append({
                'c2w': c2w,
                'image_path': image_path,
                'image_rgb': image_rgb,
                'error': reprojection_error,
                'filename': os.path.basename(image_path)
            })
            print(f"Pose estimated with error: {reprojection_error:.2f})")
        else:
            print(f"Could not estimate pose")
    
    if len(pose_candidates) == 0:
        raise ValueError("No camera poses estimated")

    pose_candidates.sort(key=lambda x: x['error'])
    selected_poses = pose_candidates
    c2w_list = [pose['c2w'] for pose in selected_poses]
    image_paths = [pose['image_path'] for pose in selected_poses]
    images_rgb = [pose['image_rgb'] for pose in selected_poses]

    return c2w_list, image_paths, images_rgb


def analyze_camera_poses(c2w_list):
    positions = np.array([c2w[:3, 3] for c2w in c2w_list])
    distances = np.linalg.norm(positions, axis=1)
    scene_extent = np.max(positions.max(axis=0) - positions.min(axis=0))
    suggested_scale = scene_extent / 10.0
    
    return suggested_scale


def visualize_with_viser(c2w_list, images_rgb, camera_matrix, save_poses=True):
    suggested_scale = analyze_camera_poses(c2w_list)
    server = viser.ViserServer(share=False)
    H, W = images_rgb[0].shape[:2]

    fov = 2 * np.arctan2(H / 2, camera_matrix[0, 0])
    aspect = W / H
    frustum_scale = max(0.01, min(suggested_scale, 0.1))

    server.scene.add_frame(
        "/origin",
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
        axes_length=frustum_scale * 3,
        axes_radius=frustum_scale * 0.1
    )

    actual_tag_size = 0.091
    scaled_tag_size = actual_tag_size * scale

    server.scene.add_box(
        "/aruco_tag",
        dimensions=(scaled_tag_size, scaled_tag_size, 0.001),
        position=np.array([0.0, 0.0, 0.0]), 
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        color=(255, 255, 0)
    )
    
    for i, (c2w, img) in enumerate(zip(c2w_list, images_rgb)):
        R = c2w[:3, :3]
        position = c2w[:3, 3]
        wxyz = viser.transforms.SO3.from_matrix(R).wxyz
        img_small = cv2.resize(img, (W//4, H//4))
        
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            wxyz=wxyz,
            position=position,
            image=img_small
        )

    if save_poses:
        poses_dict = {
            'c2w_matrices': np.array(c2w_list),
            'camera_matrix': camera_matrix,
            'num_images': len(c2w_list)
        }
        np.savez('poses.npz', **poses_dict)

    print("URL: http://localhost:8080")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nVisualization closed")


def main():
    object_image_folder = './Proj4/Media/Object'
    calibration_file = './Proj4/Code/calibration.npz'
    tag_size = 0.091
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    
    try:
        camera_matrix, dist_coeffs = load_calibration(calibration_file)
        
        c2w_list, image_paths, images_rgb = process_object_images(
            object_image_folder,
            camera_matrix,
            dist_coeffs,
            tag_size=tag_size,
            aruco_dict_type=aruco_dict_type
        )
        visualize_with_viser(c2w_list, images_rgb, camera_matrix, save_poses=True)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()