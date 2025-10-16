from feature_detect import extract_feature_descriptors, test_descriptors
import numpy as np
import cv2
import matplotlib.pyplot as plt


def test_descriptor_statistics(descriptors):
    """
    Test if descriptors are properly normalized.
    After bias/gain normalization, descriptors should have:
    - Mean ≈ 0 (close to zero)
    - Std ≈ 1 (close to one)
    """
    print("\n" + "="*60)
    print("DESCRIPTOR STATISTICS TEST")
    print("="*60)
    
    # Overall statistics
    overall_mean = np.mean(descriptors)
    overall_std = np.std(descriptors)
    
    print(f"Overall mean: {overall_mean:.6f} (should be ≈ 0)")
    print(f"Overall std:  {overall_std:.6f} (should be ≈ 1)")
    
    # Per-descriptor statistics
    per_desc_means = np.mean(descriptors, axis=1)
    per_desc_stds = np.std(descriptors, axis=1)
    
    print(f"\nPer-descriptor means: {per_desc_means.mean():.6f} ± {per_desc_means.std():.6f}")
    print(f"Per-descriptor stds:  {per_desc_stds.mean():.6f} ± {per_desc_stds.std():.6f}")
    
    # Check if any descriptors are degenerate (all same value)
    degenerate = np.sum(per_desc_stds < 0.01)
    print(f"\nDegenerate descriptors (std < 0.01): {degenerate}/{len(descriptors)}")
    
    # Value range
    print(f"\nValue range: [{descriptors.min():.2f}, {descriptors.max():.2f}]")
    print(f"  (normalized values typically in [-3, 3])")
    
    # PASS/FAIL
    passed = True
    if abs(per_desc_means.mean()) > 1e-10:
        print("\n⚠️  WARNING: Per-descriptor means not close to 0")
        passed = False
    if abs(per_desc_stds.mean() - 1.0) > 0.1:
        print("\n⚠️  WARNING: Per-descriptor stds not close to 1")
        passed = False
    if degenerate > 0:
        print(f"\n⚠️  WARNING: {degenerate} degenerate descriptors found")
        passed = False
    
    if passed:
        print("\n✅ All descriptor statistics look good!")
    
    return passed

def test_visual_inspection(image, coords, descriptors, num_to_show=15):
    """
    Visually inspect if descriptors capture meaningful patterns.
    Good descriptors should show:
    - Variation in intensity
    - Structure (edges, corners, texture)
    - Different patterns across different locations
    """
    print("\n" + "="*60)
    print("VISUAL INSPECTION TEST")
    print("="*60)
    print("Look for:")
    print("  ✓ Descriptors show texture/structure (not uniform)")
    print("  ✓ Different locations → different descriptor patterns")
    print("  ✓ Patches are centered on corners/features")
    
    num_to_show = min(num_to_show, coords.shape[1])
    indices = np.linspace(0, coords.shape[1] - 1, num_to_show, dtype=int)
    
    fig, axes = plt.subplots(2, num_to_show, figsize=(2*num_to_show, 5))
    
    for idx, i in enumerate(indices):
        # Show 8×8 descriptor patch
        ax = axes[0, idx]
        desc_patch = descriptors[i].reshape(8, 8)
        im = ax.imshow(desc_patch, cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title(f'{i}', fontsize=10)
        ax.axis('off')
        
        # Show location in image
        ax = axes[1, idx]
        y, x = coords[:, i]
        margin = 50
        y0 = max(0, int(y - margin))
        y1 = min(image.shape[0], int(y + margin))
        x0 = max(0, int(x - margin))
        x1 = min(image.shape[1], int(x + margin))
        
        patch = image[y0:y1, x0:x1]
        ax.imshow(patch, cmap='gray')
        
        local_y = y - y0
        local_x = x - x0
        ax.plot(local_x, local_y, 'r+', markersize=15, markeredgewidth=3)
        ax.axis('off')
    
    plt.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Visual inspection complete")
    print("  Do the descriptors look reasonable?")

def test_descriptor_repeatability(image_path, coord_y, coord_x, num_trials=10):
    """
    Test if extracting descriptor from same location gives similar results.
    Add small noise to position and see if descriptors remain similar.
    """
    print("\n" + "="*60)
    print("REPEATABILITY TEST")
    print("="*60)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract descriptor at exact location
    coords = np.array([[coord_y], [coord_x]])
    desc_orig, _ = extract_feature_descriptors(img, coords)
    
    if len(desc_orig) == 0:
        print("❌ Could not extract descriptor at this location")
        return False
    
    desc_orig = desc_orig[0]
    
    # Extract descriptors with small position perturbations
    distances = []
    for trial in range(num_trials):
        # Add noise: ±0.5 pixels
        noise_y = np.random.uniform(-0.5, 0.5)
        noise_x = np.random.uniform(-0.5, 0.5)
        
        coords_noisy = np.array([[coord_y + noise_y], [coord_x + noise_x]])
        desc_noisy, _ = extract_feature_descriptors(img, coords_noisy)
        
        if len(desc_noisy) > 0:
            distance = np.linalg.norm(desc_orig - desc_noisy[0])
            distances.append(distance)
    
    if len(distances) == 0:
        print("❌ Failed to extract perturbed descriptors")
        return False
    
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    
    print(f"Position: ({coord_x}, {coord_y})")
    print(f"Perturbation: ±0.5 pixels × {num_trials} trials")
    print(f"Mean descriptor distance: {mean_dist:.4f}")
    print(f"Max descriptor distance:  {max_dist:.4f}")
    print(f"  (should be < 1.0 for good repeatability)")
    
    passed = mean_dist < 1.0 and max_dist < 2.0
    
    if passed:
        print("\n✅ Repeatability test PASSED")
    else:
        print("\n⚠️  Repeatability test FAILED - descriptors too sensitive to position")
    
    return passed

def test_descriptor_distinctiveness(descriptors, num_samples=100):
    """
    Test if different features have different descriptors.
    Measure pairwise distances - should be reasonably large.
    """
    print("\n" + "="*60)
    print("DISTINCTIVENESS TEST")
    print("="*60)
    
    if len(descriptors) < 2:
        print("❌ Need at least 2 descriptors")
        return False
    
    # Sample pairs to avoid computing all pairwise distances
    num_samples = min(num_samples, len(descriptors) * (len(descriptors) - 1) // 2)
    
    distances = []
    for _ in range(num_samples):
        i, j = np.random.choice(len(descriptors), 2, replace=False)
        dist = np.linalg.norm(descriptors[i] - descriptors[j])
        distances.append(dist)
    
    distances = np.array(distances)
    
    print(f"Pairwise distances (n={num_samples} pairs):")
    print(f"  Mean:   {distances.mean():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Min:    {distances.min():.4f}")
    print(f"  Max:    {distances.max():.4f}")
    print(f"\nGood descriptors: mean > 3.0, min > 0.5")
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Count')
    plt.title('Distribution of Pairwise Descriptor Distances')
    plt.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.2f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    passed = distances.mean() > 3.0 and distances.min() > 0.5
    
    if passed:
        print("\n✅ Distinctiveness test PASSED")
    else:
        print("\n⚠️  Descriptors may not be distinctive enough")
    
    return passed

def run_all_descriptor_tests(image_path, num_features=500):
    """
    Run all descriptor tests.
    """
    import os
    
    print("\n" + "#"*60)
    print("FEATURE DESCRIPTOR TEST SUITE")
    print("#"*60)
    
    # Extract features
    coords, descriptors, img_gray, img_rgb = test_descriptors(
        image_path, 
        num_features=num_features
    )
    
    if descriptors.shape[0] == 0:
        print("\n❌ FAILED: No descriptors extracted")
        return False
    
    # Run tests
    results = {}
    
    # Test 1: Statistics
    results['statistics'] = test_descriptor_statistics(descriptors)
    
    # Test 2: Visual inspection
    test_visual_inspection(img_rgb, coords, descriptors, num_to_show=12)
    
    # Test 3: Repeatability (pick a corner in the middle)
    h, w = img_gray.shape
    test_y, test_x = h // 2, w // 2
    results['repeatability'] = test_descriptor_repeatability(
        image_path, test_y, test_x, num_trials=10
    )
    
    # Test 4: Distinctiveness
    results['distinctiveness'] = test_descriptor_distinctiveness(descriptors)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Descriptors look good.")
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
    
    return all_passed


# Run the complete test suite
if __name__ == "__main__":
    base_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_1'
    
    success = run_all_descriptor_tests(
        f'{base_path}/DSC_0305.jpg',
        num_features=500
    )