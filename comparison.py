import cv2
import numpy as np

def align_images(reference_img, sample_img):
    """
    Aligns the sample image to the reference image using ORB feature matching.
    Returns: aligned_sample, status (True/False)
    """
    # 1. Convert to grayscale
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    smpl_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect ORB features and compute descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(smpl_gray, None)
    
    if des1 is None or des2 is None:
        return sample_img, False
        
    # 3. Match features using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    # 4. Sort matches by score/distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top 15% of matches
    keep = int(len(matches) * 0.15)
    matches = matches[:keep]
    
    if len(matches) < 4:
        return sample_img, False
        
    # 5. Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
        
    # 6. Find Homography matrix
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # 7. Use homography to warp sample image
    height, width, channels = reference_img.shape
    aligned_sample = cv2.warpPerspective(sample_img, h, (width, height))
    
    return aligned_sample, True

def detect_differences(reference_img, aligned_sample):
    """
    Finds physical differences between aligned images.
    Returns: diff_mask, diff_visual
    """
    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    smpl_gray = cv2.cvtColor(aligned_sample, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    smpl_blur = cv2.GaussianBlur(smpl_gray, (5, 5), 0)
    
    # Absolute difference
    diff = cv2.absdiff(ref_blur, smpl_blur)
    
    # Thresholding to get binary mask
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Generate visualization
    # Highlight differences in red on the aligned sample
    diff_visual = aligned_sample.copy()
    diff_visual[thresh == 255] = [0, 0, 255] # Red for differences
    
    return thresh, diff_visual
