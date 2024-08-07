import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load images
im1 = cv2.imread("C:/Users/rites/Downloads/image1.jpeg")
im2 = cv2.imread("C:/Users/rites/Downloads/image2.jpeg")

# Convert to grayscale
img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create(50)

# Find keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors
matches = matcher.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None)

# Display the matching image
cv2.imshow("Matches image", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
