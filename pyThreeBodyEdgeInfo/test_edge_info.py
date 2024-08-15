import pyBodyEdgeInfo as pybei
import numpy as np
import time

kokkos_obj = pybei.KokkosInterface()

data = np.loadtxt('data_sample.txt', dtype="float32")
print(data.shape)

edgeInfo = pybei.BodyEdgeInfo()
edgeInfo.setTwoBodyEpsilon(5.0)
edgeInfo.setThreeBodyEpsilon(5.0)

edgeInfo.setTargetSites(data)

edgeInfo.buildTwoBodyEdgeInfo()
edgeInfo.buildThreeBodyEdgeInfo()

N = 5
start = time.time()
for i in range(N):
    edgeInfo.setTargetSites(data)

    edgeInfo.buildTwoBodyEdgeInfo()
    edgeInfo.buildThreeBodyEdgeInfo()

    threeBodyEdgeInfo = edgeInfo.getThreeBodyEdgeInfo()
    threeBodyEdgeSelfInfo = edgeInfo.getThreeBodyEdgeSelfInfo()
    twoBodyEdgeInfo = edgeInfo.getTwoBodyEdgeInfo()
    morseForce = edgeInfo.getMorseForce()

end = time.time()
print("{:8.6f}".format((end - start) / N) + 's')
print(threeBodyEdgeInfo.shape)
print(threeBodyEdgeSelfInfo.shape)
print(twoBodyEdgeInfo.shape)
print(morseForce.shape)

N = 5
start = time.time()
for i in range(N):
    edgeInfo.setTargetSites(data)

    edgeInfo.buildTwoBodyEdgeInfo()
    edgeInfo.buildThreeBodyEdgeInfo()

    for j in range(5):
        indexRange = np.arange(25000, dtype=np.int64) + j * 25000
        threeBodyEdgeInfo = edgeInfo.getThreeBodyEdgeInfoByIndex(indexRange)
        threeBodyEdgeSelfInfo = edgeInfo.getThreeBodyEdgeSelfInfoByIndex(
            indexRange)
        # twoBodyEdgeInfo = edgeInfo.getTwoBodyEdgeInfoByIndex(indexRange)
        # morseForce = edgeInfo.getMorseForceByIndex(indexRange)


end = time.time()
print("{:8.6f}".format((end - start) / N) + 's')

N = 5
start = time.time()
for i in range(N):
    edgeInfo.setTargetSites(data)

    edgeInfo.buildTwoBodyEdgeInfo()
    edgeInfo.buildThreeBodyEdgeInfo()

    length1 = 0
    length2 = 0
    length3 = 0
    for j in range(10):
        indexRange = np.arange(12500, dtype=np.int64) + j * 12500
        threeBodyEdgeInfo = edgeInfo.getThreeBodyEdgeInfoByIndex(indexRange)
        length1 += threeBodyEdgeInfo.shape[1]
        threeBodyEdgeSelfInfo = edgeInfo.getThreeBodyEdgeSelfInfoByIndex(
            indexRange)
        length2 += threeBodyEdgeSelfInfo.shape[1]
        # twoBodyEdgeInfo = edgeInfo.getTwoBodyEdgeInfoByIndex(indexRange)
        # length3 += twoBodyEdgeInfo.shape[1]
        # morseForce = edgeInfo.getMorseForceByIndex(indexRange)

end = time.time()
print("{:8.6f}".format((end - start) / N) + 's')
print(length1)
print(length2)
# print(length3)
