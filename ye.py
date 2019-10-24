import cv2
import os
import numpy
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

os.chdir("C:/Users/Anurag/Desktop/intern/python-fingerprint-recognition-master")
from enhance import image_enhance

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)
    
    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    #print(W)
    #print(H)
    
    filtersize = 6
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]
            #print (filter0)

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
      
	# Threshold
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	# Normalize to 0 and 1 range
	img[img == 255] = 1
	
	#Thinning
	skeleton = skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
	# Harris corners
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# Extract keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, des = orb.compute(img, keypoints)
     
	return (keypoints, des);


def main():
	
    img1 = cv2.imread("C:/Users/Nimesh Shahdadpuri/Desktop/DMRC Intern/database/106_1.tif" , cv2.IMREAD_GRAYSCALE)
          
    kp1, des1 = get_descriptors(img1)
    	
    #print (des1)	
    #print (des1.shape)
    img2 = cv2.imread("C:/Users/Nimesh Shahdadpuri/Desktop/DMRC Intern/database/106_2.tif" , cv2.IMREAD_GRAYSCALE)
    kp2, des2 = get_descriptors(img2)
    #print (des2)
    	
    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches= bf.match(des1,des2)
    matches = sorted(matches, key= lambda match:match.distance)
    #print (len(matches))
    # Plot keypoints
    img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
    img5 = cv2.drawKeypoints(img2, kp2, outImage=None)
    #f, axarr = plt.subplots(1,2)
    print ("First Fingerprint")
    #axarr[0].imshow(img4) 
    plt.imshow(img4)
    plt.show()
    print ("Second Fingerprint")
    #axarr[1].imshow(img5)
    plt.imshow(img5)
    plt.show()
    # Plot matches
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    print ("All the matching points and the corresponding distances")
    plt.imshow(img3)
    plt.show()
    	
    # Calculate score
    score = 0
    for match in matches:
        score += match.distance
    score_threshold = 40
    matchper= score/len(matches)
    print(matchper)     
          
    if  matchper < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")
	
	
	
if _name_ == "_main_":
	try: 
		main()
	except:
		raise