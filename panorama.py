""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import cv2

debug_count = 0


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    
    # print("Image shape= ",image.shape)

    min_x = 0
    min_y = 0

    max_x = image.shape[1]
    max_y = image.shape[0]

    corners[0,0] = [min_x,min_y]
    corners[1,0] = [min_x,max_y]
    corners[2,0] = [max_x,min_y]
    corners[3,0] = [max_x,max_y]

    """
    global debug_count

    debug_image = np.copy(image)

    debug_image[min_y,min_x] = [0,0,255]
    debug_image[min_y,max_x] = [0,0,255]
    debug_image[max_y,min_x] = [0,0,255]
    debug_image[max_y,max_x] = [0,0,255]

    filename = "debug_corner_" + str(debug_count) + ".png"
    debug_count += 1
    cv2.imwrite(filename, debug_image)
    print(filename)
    """

    # print(corners)
    return corners

def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    
    match_count = len(matches)

    for i in range(match_count):

        next_match = matches[i]

        image_1_points[i,0] = image_1_kp[next_match.queryIdx].pt
        image_2_points[i,0] = image_2_kp[next_match.trainIdx].pt

    """
    print("image_1_points = ",image_1_points)
    print("\n\n")
    print("image_2_points = ",image_2_points)
    print("\n\n")
    """

    homography, mask = cv2.findHomography(image_1_points,image_2_points,method=cv2.RANSAC, ransacReprojThreshold=5.0)

    return homography

def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # print("homography")
    # print(homography)


    # print("corners 1 - in")
    # print(corners_1)

    corners_1_out = cv2.perspectiveTransform(corners_1, homography)

    corners = np.concatenate((corners_1_out,corners_2),axis=0)

    x_slice = corners[:,:,0:1]
    y_slice = corners[:,:,1:]

    min_x = np.amin(x_slice)
    max_x = np.amax(x_slice)

    min_y = np.amin(y_slice)
    max_y = np.amax(y_slice)

    min_coord = np.array(([min_x,min_y]),dtype=np.float64)
    max_coord = np.array(([max_x, max_y]),dtype=np.float64)

    return min_coord, max_coord


def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE

    translation_matrix = [[1, 0, -min_xy[0]],
                          [0, 1, -min_xy[1]],
                          [0, 0, 1]]

    transform_matrix = np.dot(translation_matrix, homography)

    # image_out = np.zeros(canvas_size).astype(np.uint8)

    image_out = cv2.warpPerspective(image, transform_matrix, dsize=canvas_size)

    return image_out

def drawMatchesBetweenImages(image_1, kp_1, image_2, kp_2, matches):

    # print(kp_1)
    image_out = np.zeros_like(image_1)
    image_out = cv2.drawMatches(image_1, kp_1, image_2, kp_2, matches, image_out)
    cv2.imwrite("draw_matches_out.png", image_out)

def cropPanoramaHeight(min_y, max_y, corners_1, corners_2, homography, panorama):

    # print("orig corners",corners_1)

    # prev_min_y = np.amin(corners_1[:,:,1:])
    # prev_max_y = np.amax(corners_1[:,:,1:])

    # diff_min_y = np.rint(np.absolute(prev_min_y - min_y)).astype(np.intc) + 1
    # diff_max_y = np.rint(np.absolute(prev_max_y - max_y)).astype(np.intc) -1
    # diff_max_y *= -1

    # print("diff_min_y",diff_min_y)
    # print("diff_max_y",diff_max_y)

    # print(panorama.shape)

    image_out = np.copy(panorama)

    #Crop sides
    corners_1_warped = cv2.perspectiveTransform(corners_1, homography)

    left_crop = np.absolute(corners_1_warped[0][0][0] - corners_1_warped[1][0][0]).astype(np.intc)
    image_out = image_out[:,left_crop:,:]

    right_crop = np.absolute(corners_2[2][0][0] - corners_2[3][0][0]).astype(np.intc)
    right_crop *= -1

    if right_crop == 0:
        image_out = image_out[:,:-1,:]
    else:
        image_out = image_out[:,:right_crop,:]

    # print("left_crop",left_crop)
    # print("right_crop",right_crop)

    # cv2.imwrite("post_LR_crop.png",image_out


    #Crop from the top     ##################################

    max_crop = 0

    for i in range(image_out.shape[1]):

        if np.sum(image_out[0][i]) == 0:

            crop_length = 0
            for j in range(1,int(image_out.shape[0]/2)):
                
                if np.sum(image_out[j][i]) == 0:
                    crop_length += 1
                else:
                    break

            if max_crop < crop_length:
                max_crop = crop_length

    image_out = image_out[max_crop:,:,:]

    print("top crop=",max_crop)

    #Crop from the bottom     ##################################

    max_crop = 0
    
    for i in range(image_out.shape[1]):

        if np.sum(image_out[image_out.shape[0]-1][(image_out.shape[1]-1)-i]) == 0:

            crop_length = 0
            for j in range(1,int(image_out.shape[0]/2)):
                
                if np.sum(image_out[(image_out.shape[0]-1)-j][(image_out.shape[1]-1)-i]) == 0:
                    crop_length += 1
                else:
                    break

            if max_crop < crop_length:
                max_crop = crop_length

    max_crop *= -1

    if max_crop < 0:
        image_out = image_out[:max_crop,:,:]

    print("bottom crop=",max_crop)

    return image_out

# def findMinSeam():



def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

        You MUST replace the basic insertion blend provided here to earn
                         credit for this function.

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

           Be creative -- good blending is the primary way to earn
                  Above & Beyond credit on this assignment.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        (3) You can modify this function however you see fit -- e.g., change
        input parameters, return values, etc. -- to develop your blending
        process.
    """
    kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2, num_matches)
    # drawMatchesBetweenImages(image_1, kp1, image_2, kp2, matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    output_image = warpCanvas(image_1, homography, min_xy, max_xy)

    cv2.imwrite("warped_out.png",output_image)

    print("min_xy",min_xy)
    print("max_xy",max_xy)


    # WRITE YOUR CODE HERE - REPLACE THIS WITH YOUR BLENDING CODE
    min_xy = min_xy.astype(np.int)
    output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],-min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2
    cv2.imwrite("pre_crop.png",output_image)
    output_image = cropPanoramaHeight(min_xy[1], max_xy[1], corners_1, corners_2, homography, output_image)

    return output_image
    # return None
    # END OF FUNCTION
