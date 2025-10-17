from feature_detect import test_harris_anms, extract_feature_descriptors, match_image_pair
from weighted_avg import computeH, test_canvas_placement, rectify_image_example, create_panorama_from_data, load_panorama_data
from laplacian import create_panorama_from_data_pyramid
from ransac import create_automatic_panorama
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    #warped_nn, warped_bil, mask_nn, mask_bil = rectify_image_example()
    base_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_1'
    image_numbers = [305,306,307,308,309,310]

    # Weighted Average Blending
    #panorama = create_panorama_from_data(base_path, image_numbers)

    # Laplacian Pyramid Blending
    panorama = create_panorama_from_data_pyramid(base_path, image_numbers, output_name='panorama_0305_to_0310_pyramid')

    # Automatic Panorama
    '''panorama, homographies = create_automatic_panorama(
        base_path, image_numbers,
        num_features=1000,
        ratio_threshold=0.5,
        ransac_iters=5000,
        ransac_threshold=4.0,
        pyramid_levels=4,
        output_name='automatic_panorama_0865_to_0866'
    )'''