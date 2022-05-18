import time

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
    # print(flow)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    np.save(flowMap, mag)
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)
    # print(np.load(flowMap))


def createClowMaps(startInd, endInd):
    for i in range(startInd, endInd+1):
        num = str(i)
        file1 = str("0"*(6-len(num))) + num
        path1 = "image_2/" + file1 + ".png"
        path2 = "image_3/" + file1 + ".png"
        print(path1)
        print(path2)
        print("______________")
        lucus(path1, path2, "FlowMap/flowMapV" + str(i) + ".npy")

def compare(disp, groundTruth, k):
    d = np.load(disp)
    gt = np.load(groundTruth)
    num = 0
    den = 0
    for x in range(len(d)):
        for y in range(len(d[x])):
            pxd = d[x][y]
            pxgt = gt[x][y]
            if not pxd == 0 and not pxgt <= 0:
                if (pxd - pxgt)**2 > k**2:
                    num += 1
                den += 1
    return num, den

def compareAll(start, end, k):
    num = 0
    den = 0
    for i in range(start, end+1):
        number = str(i)
        file1 = str("0"*(6-len(number))) + number
        gt = "GroundTruth/" + file1 + ".npy"
        d = "FlowMap/flowMapV" + number + ".npy"
        tempNum, tempDen = compare(d, gt, k)
        num += tempNum
        den += tempDen
        print(str(i) + "  " + str(num/den))
    print(num/den)
    return num/den

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compareAll(0, 7480, 1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
