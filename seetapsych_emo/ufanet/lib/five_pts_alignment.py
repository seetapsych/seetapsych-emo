from .face_align import crop_5pts_vipl_256   # import the function crop_align

def face_align_crop(image, keypoints):
    '''
    image: numpy.ndarray, uint8, [0,255], RGB, (H,W,3), raw image after reading
    keypoints: 5x2 np.array,
    facial landmark coordinates, each row is a pair of coordinates (x, y)
    landmark coordinates in order: left eye center, right eye center, nose tip, left mouth corner, right mouth corner
    '''

    keypoints = keypoints.astype(int)
    aligned_frame = crop_5pts_vipl_256.crop_align(image, keypoints)
    return aligned_frame #return the numpy array of the aligned face
