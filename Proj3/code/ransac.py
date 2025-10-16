from laplacian import computeH, create_panorama_sequential_pyramid, create_panorama_from_data_pyramid
from feature_detect import test_descriptors, match_features
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.ndimage import distance_transform_edt

def compute_homography_ransac(matches, coords1, coords2, n_iters=5000, threshold=3.0):
    if len(matches) < 4:
        print("ERROR: Need at least 4 matches")
        return None, None, 0
    
    pts1 = []
    pts2 = []
    for i, j in matches:
        y1, x1 = coords1[:, i]
        y2, x2 = coords2[:, j]
        pts1.append([x1, y1])
        pts2.append([x2, y2])
    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    
    best_inliers = None
    best_num_inliers = 0
    H_best = None

    for iteration in range(n_iters):
        indices = np.random.choice(len(matches), 4, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        try:
            H = computeH(sample_pts1, sample_pts2)
        except:
            continue
   
        pts1_homogeneous = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_predicted = (H @ pts1_homogeneous.T).T
        pts2_predicted = pts2_predicted[:, :2] / pts2_predicted[:, 2:3]
        
        distances = np.linalg.norm(pts2 - pts2_predicted, axis=1)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            H_best = H

    if best_inliers is not None and best_num_inliers >= 4:
        inlier_pts1 = pts1[best_inliers]
        inlier_pts2 = pts2[best_inliers]
        H_best = computeH(inlier_pts1, inlier_pts2)

    return H_best, best_inliers, best_num_inliers

def create_automatic_panorama(base_path, image_numbers, 
                              num_features=500, ratio_threshold=0.8,
                              ransac_iters=5000, ransac_threshold=3.0,
                              pyramid_levels=4, output_name='auto_panorama'):

    images = []
    for num in image_numbers:
        img_path = os.path.join(base_path, f'DSC_0{num}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

    homographies = []
    
    for i in range(len(image_numbers) - 1):
        num1 = image_numbers[i]
        num2 = image_numbers[i + 1]
        img_path1 = os.path.join(base_path, f'DSC_0{num1}.jpg')
        img_path2 = os.path.join(base_path, f'DSC_0{num2}.jpg')

        coords1, desc1, gray1, rgb1 = test_descriptors(img_path1, num_features)
        coords2, desc2, gray2, rgb2 = test_descriptors(img_path2, num_features)

        matches, distances = match_features(desc1, desc2, ratio_threshold)
        print(f"  Found {len(matches)} matches")
        
        if len(matches) < 4:
            raise ValueError(f"Not enough matches for pair {num1}-{num2}")

        H, inliers, num_inliers = compute_homography_ransac(
            matches, coords1, coords2,
            n_iters=ransac_iters,
            threshold=ransac_threshold
        )
        
        if H is None:
            raise ValueError(f"RANSAC failed for pair {num1}-{num2}")
        
        homographies.append(H)
    
    panorama = create_panorama_sequential_pyramid(images, homographies, pyramid_levels)

    output_path = os.path.join(base_path, f'{output_name}.png')
    plt.figure(figsize=(24, 8))
    plt.imshow(panorama)
    plt.axis('off')
    plt.title(f'Automatic Panorama (RANSAC): DSC_0{image_numbers[0]} to DSC_0{image_numbers[-1]}')
    plt.tight_layout()
    #plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return panorama, homographies

if __name__ == "__main__":
    base_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_3'
    image_numbers = [865,866]
    panorama, homographies = create_automatic_panorama(
        base_path, image_numbers,
        num_features=1000,
        ratio_threshold=0.5,
        ransac_iters=5000,
        ransac_threshold=4.0,
        pyramid_levels=4,
        output_name='automatic_panorama_0865_to_0866'
    )