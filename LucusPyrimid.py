import numpy as np
import cv2 as cv
import imageio as iio

def lucus(img1, img2, flowMap):
    # params for corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # lucas parameters for algorithm
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                               10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # bring in lhs and rhs
    lhs = iio.imread(img1)
    rhs = iio.imread(img2)

    lhs_gray = cv.cvtColor(lhs, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(lhs_gray, mask=None, **feature_params)
    mask = np.zeros_like(lhs)

    rhs_gray = cv.cvtColor(rhs, cv.COLOR_BGR2GRAY)

    lhs = iio.imread(img1)
    rhs = iio.imread(img2)
    prvs = cv.cvtColor(lhs, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(lhs)
    hsv[..., 1] = 255
    next = cv.cvtColor(rhs, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    cv.imwrite(flowMap, gray)


def createClowMaps(n):
    writeIter = 0
    for i in range(n):
        num = str(i)
        file1 = str("0"*(6-len(num))) + num + "_10"
        file2 = str("0"*(6-len(num))) + num + "_11"
        path1 = "image_2/" + file1 + ".png"
        path2 = "image_3/" + file1 + ".png"
        path3 = "image_2/" + file2 + ".png"
        path4 = "image_3/" + file2 + ".png"
        print(path1)
        print(path2)
        print("______________")
        lucus(path1, path2, "FlowMap/flowMapV" + str(writeIter) + ".png")
        writeIter += 1
        lucus(path3, path4, "FlowMap/flowMapV" + str(writeIter) + ".png")
        print(path3)
        print(path4)
        print("______________")
        writeIter += 1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    createClowMaps(200)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
