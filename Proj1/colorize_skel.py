# CS194-26 (CS294-26): Project 1 - OpenCV Feature-Based Homography Alignment
# Following the exact approach from: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2

def alignImages(im1, im2):
    """
    Align im1 to im2 using feature-based homography
    Following the OpenCV tutorial exactly
    """
    # Convert images to grayscale for OpenCV
    if len(im1.shape) == 3:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    else:
        im1Gray = im1
        
    if len(im2.shape) == 3:
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    else:
        im2Gray = im2

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches (optional, for debugging)
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography to warp image
    height, width = im2.shape[:2]
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h

def process_image_opencv(imname):
    """
    Process Prokudin-Gorskii image using OpenCV homography alignment
    Following skeleton code pattern with sk.img_as_float
    """
    print(f"Processing with OpenCV homography: {imname}")
    
    # Read in the image (following skeleton)
    im = skio.imread(imname)
    print(f"Original image shape: {im.shape}, dtype: {im.dtype}, range: [{im.min()}, {im.max()}]")
    
    # Convert to double (following skeleton)
    im = sk.img_as_float(im)
    print(f"After sk.img_as_float: shape {im.shape}, dtype: {im.dtype}, range: [{im.min():.3f}, {im.max():.3f}]")
    
    # Compute the height of each part (following skeleton)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    
    # Separate color channels (following skeleton BGR order)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    print(f"Each channel shape: {b.shape}, dtype: {b.dtype}")
    
    # Use blue channel as reference (following skeleton pattern)
    reference = b
    
    # Align green channel to blue
    print("\nAligning green channel...")
    try:
        ag, hGreen = alignImages(g, reference)
        
        if np.allclose(hGreen, np.eye(3)):
            print("Green alignment returned identity matrix - alignment failed")
            g_dx, g_dy = 0, 0
        else:
            # Extract translation from homography
            g_dx = hGreen[0, 2]
            g_dy = hGreen[1, 2]
            print(f"Green homography translation: dx={g_dx:.2f}, dy={g_dy:.2f}")
        
    except Exception as e:
        print(f"Green alignment failed: {e}")
        ag = g
        g_dx, g_dy = 0, 0
    
    # Align red channel to blue
    print("\nAligning red channel...")
    try:
        ar, hRed = alignImages(r, reference)
        
        if np.allclose(hRed, np.eye(3)):
            print("Red alignment returned identity matrix - alignment failed")
            r_dx, r_dy = 0, 0
        else:
            # Extract translation from homography
            r_dx = hRed[0, 2]
            r_dy = hRed[1, 2]
            print(f"Red homography translation: dx={r_dx:.2f}, dy={r_dy:.2f}")
        
    except Exception as e:
        print(f"Red alignment failed: {e}")
        ar = r
        r_dx, r_dy = 0, 0
    
    # Create a color image (following skeleton)
    im_out = np.dstack([ar, ag, b])
    
    print(f"\nOpenCV Homography Results:")
    print(f"Green displacement: ({g_dx:.1f}, {g_dy:.1f})")
    print(f"Red displacement: ({r_dx:.1f}, {r_dy:.1f})")
    
    # Save the image (following skeleton pattern)
    fname = f'output_opencv_{imname.split("/")[-1].replace(".tif", ".jpg")}'
    skio.imsave(fname, im_out)
    print(f"Saved: {fname}")
    
    # Display the image (following skeleton)
    plt.figure(figsize=(12, 8))
    plt.imshow(im_out)
    plt.title(f'OpenCV Homography Alignment\nG: ({g_dx:.1f}, {g_dy:.1f}), R: ({r_dx:.1f}, {r_dy:.1f})')
    plt.axis('off')
    plt.show()
    
    return im_out, (g_dx, g_dy), (r_dx, r_dy)

def compare_with_simple_translation(imname):
    """
    Compare homography results with simple translation extraction
    """
    print(f"\n=== COMPARING HOMOGRAPHY VS SIMPLE TRANSLATION ===")
    
    # Read and prepare image
    im = skio.imread(imname)
    
    # Handle uint16 properly
    if im.dtype == np.uint16:
        im = (im / 256).astype(np.uint8)
    elif im.dtype == np.float64 or im.dtype == np.float32:
        im = (im * 255).astype(np.uint8)
    elif im.max() <= 1.0:
        im = (im * 255).astype(np.uint8)
    
    height = im.shape[0] // 3
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    # Method 1: Full homography alignment (as above)
    print("\n1. Full Homography Alignment:")
    try:
        greenAligned, hGreen = alignImages(g, b)
        redAligned, hRed = alignImages(r, b)
        
        # Check if alignment actually worked
        if np.allclose(hGreen, np.eye(3)) or np.allclose(hRed, np.eye(3)):
            print("   Homography alignment failed (returned identity matrix)")
            return
        
        g_dx_homo = hGreen[0, 2]
        g_dy_homo = hGreen[1, 2]
        r_dx_homo = hRed[0, 2]
        r_dy_homo = hRed[1, 2]
        
        print(f"   Green: ({g_dx_homo:.1f}, {g_dy_homo:.1f})")
        print(f"   Red: ({r_dx_homo:.1f}, {r_dy_homo:.1f})")
        
        # Method 2: Extract just translation and apply with np.roll
        print("\n2. Translation-only (using np.roll):")
        
        # Convert back to float for np.roll
        b_float = b.astype(np.float32) / 255.0
        g_float = g.astype(np.float32) / 255.0
        r_float = r.astype(np.float32) / 255.0
        
        # Apply translation using np.roll
        g_translated = np.roll(g_float, int(round(g_dx_homo)), axis=1)
        g_translated = np.roll(g_translated, int(round(g_dy_homo)), axis=0)
        
        r_translated = np.roll(r_float, int(round(r_dx_homo)), axis=1)
        r_translated = np.roll(r_translated, int(round(r_dy_homo)), axis=0)
        
        # Create comparison images
        homo_result = np.dstack([redAligned/255.0, greenAligned/255.0, b/255.0])
        translation_result = np.dstack([r_translated, g_translated, b_float])
        
        # Display both results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        ax1.imshow(homo_result)
        ax1.set_title(f'Homography Warping\nG: ({g_dx_homo:.1f}, {g_dy_homo:.1f}), R: ({r_dx_homo:.1f}, {r_dy_homo:.1f})')
        ax1.axis('off')
        
        ax2.imshow(translation_result)
        ax2.set_title(f'Simple Translation\nG: ({g_dx_homo:.1f}, {g_dy_homo:.1f}), R: ({r_dx_homo:.1f}, {r_dy_homo:.1f})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"   Translation displacements: G({int(round(g_dx_homo))}, {int(round(g_dy_homo))}), R({int(round(r_dx_homo))}, {int(round(r_dy_homo))})")
        
    except Exception as e:
        print(f"   Failed: {e}")
        print("   Trying fallback correlation-based alignment...")
        
        # Fallback to correlation-based alignment for comparison
        try:
            from skimage.filters import sobel
            
            def simple_align(ch1, ch2, search_range=20):
                best_score = -1
                best_disp = (0, 0)
                
                # Use edge-based alignment
                ch1_edges = sobel(ch1.astype(np.float32) / 255.0)
                ch2_edges = sobel(ch2.astype(np.float32) / 255.0)
                
                margin = 100
                ref_inner = ch2_edges[margin:-margin, margin:-margin]
                
                for dx in range(-search_range, search_range + 1):
                    for dy in range(-search_range, search_range + 1):
                        shifted = np.roll(ch1_edges, dx, axis=1)
                        shifted = np.roll(shifted, dy, axis=0)
                        shifted_inner = shifted[margin:-margin, margin:-margin]
                        
                        # NCC
                        ref_flat = ref_inner.flatten()
                        shifted_flat = shifted_inner.flatten()
                        ref_centered = ref_flat - np.mean(ref_flat)
                        shifted_centered = shifted_flat - np.mean(shifted_flat)
                        
                        numer = np.dot(ref_centered, shifted_centered)
                        denom = np.linalg.norm(ref_centered) * np.linalg.norm(shifted_centered)
                        
                        if denom > 0:
                            score = numer / denom
                            if score > best_score:
                                best_score = score
                                best_disp = (dx, dy)
                
                return best_disp, best_score
            
            print("   Using edge-based correlation alignment:")
            g_disp, g_score = simple_align(g, b)
            r_disp, r_score = simple_align(r, b)
            
            print(f"   Green: {g_disp}, score: {g_score:.4f}")
            print(f"   Red: {r_disp}, score: {r_score:.4f}")
            
            # Create result with correlation alignment
            b_float = b.astype(np.float32) / 255.0
            g_float = g.astype(np.float32) / 255.0
            r_float = r.astype(np.float32) / 255.0
            
            g_aligned = np.roll(g_float, g_disp[0], axis=1)
            g_aligned = np.roll(g_aligned, g_disp[1], axis=0)
            r_aligned = np.roll(r_float, r_disp[0], axis=1)
            r_aligned = np.roll(r_aligned, r_disp[1], axis=0)
            
            corr_result = np.dstack([r_aligned, g_aligned, b_float])
            
            plt.figure(figsize=(12, 8))
            plt.imshow(corr_result)
            plt.title(f'Edge-Based Correlation Alignment\nG: {g_disp}, R: {r_disp}')
            plt.axis('off')
            plt.show()
            
        except Exception as e2:
            print(f"   Fallback also failed: {e2}")

# Global parameters (from OpenCV tutorial)
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.30

# Example usage
if __name__ == "__main__":
    
    # Test with Emir image
    emir_path = '/Users/junwei/Fall2025/CS180/cs180 proj1 data/emir.tif'
    
    try:
        print("=== OPENCV HOMOGRAPHY ALIGNMENT ===")
        result = process_image_opencv(emir_path)
        
        print("\n=== COMPARISON WITH TRANSLATION ===")
        compare_with_simple_translation(emir_path)
        
    except FileNotFoundError:
        print(f"File not found: {emir_path}")
        print("Please update the path to your image file")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()