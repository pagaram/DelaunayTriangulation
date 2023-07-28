import numpy as np
import matplotlib.pyplot as plt

def readData(file):
    with open(file) as f:
        ind = -1
        numPts = 0
        Pts = np.empty((0, 3))
        Triangles = np.empty((0, 3))

        for line in f:
            ind = ind + 1
            data = [int(x) for x in line.split()]

            if ind == 0:
                numPts = data[0]

            elif ind <= numPts:
                Pts = np.vstack((Pts, [data[0], data[1], data[2]]))

            else:
                Triangles = np.vstack((Triangles, [data[0], data[1], data[2]]))

    return Pts, Triangles


def segmentAngle(segA, segB):
    dotProd = np.dot(segA, segB)
    scaleA = np.sqrt((segA[0]*segA[0]) + (segA[1]*segA[1]))
    scaleB = np.sqrt((segB[0] * segB[0]) + (segB[1] * segB[1]))

    angle = np.arccos(dotProd/(scaleA*scaleB))
    angle = angle * 180/np.pi

    return angle


def commonEdge(TriangleA, TriangleB):
    foundCommonEdge = False
    commonEdge = np.array([0, 0])
    commonPts = 0

    for k in range(3):
        for l in range(3):
            if TriangleA[k] == TriangleB[l]:
                commonPts += 1
                commonEdge[int(commonPts-1)] = TriangleA[k]

    if commonPts == 2:
        foundCommonEdge = True

    return foundCommonEdge, commonEdge


def findTriangles(edge, Triangles):
    TriangleIndices = np.array([0, 0])
    index = -1

    for t in range(Triangles.shape[0]):
        edgeMatch = 0
        for c in range(3):
            if Triangles[t, c] == edge[0] or Triangles[t, c] == edge[1]:
                edgeMatch += 1

            if edgeMatch == 2 and index < 1:
                index += 1
                TriangleIndices[int(index)] = t

    return TriangleIndices

def computeAngles(spots, Pts):
    angles = np.array([0, 0, 0])
    spot1 = int(spots[0])
    spot2 = int(spots[1])
    spot3 = int(spots[2])

    seg12 = np.array([Pts[spot2, 1] - Pts[spot1, 1], Pts[spot2, 2] - Pts[spot1, 2]])
    seg13 = np.array([Pts[spot3, 1] - Pts[spot1, 1], Pts[spot3, 2] - Pts[spot1, 2]])
    angles[0] = segmentAngle(seg12, seg13)

    seg21 = np.array([Pts[spot1, 1] - Pts[spot2, 1], Pts[spot1, 2] - Pts[spot2, 2]])
    seg23 = np.array([Pts[spot3, 1] - Pts[spot2, 1], Pts[spot3, 2] - Pts[spot2, 2]])
    angles[1] = segmentAngle(seg21, seg23)

    seg31 = np.array([Pts[spot1, 1] - Pts[spot3, 1], Pts[spot1, 2] - Pts[spot3, 2]])
    seg32 = np.array([Pts[spot2, 1] - Pts[spot3, 1], Pts[spot2, 2] - Pts[spot3, 2]])
    angles[2] = segmentAngle(seg31, seg32)

    return angles

def edgeExists(edge, Triangles):
    exists = False
    exInd = 0

    while exInd < Triangles.shape[0] and not exists:
        if edge[0] == Triangles[exInd, 0] or edge[0] == Triangles[exInd, 1] or edge[0] == Triangles[exInd, 2]:
            if edge[1] == Triangles[exInd, 0] or edge[1] == Triangles[exInd, 1] or edge[1] == Triangles[exInd, 2]:
                exists = True
        if not exists:
            exInd += 1

    return exists

def legalizeEdge(edge, Pts, Triangles):

    TriangleIndices = findTriangles(edge, Triangles)

    #compute vector of angles
    angleVec = np.zeros(6)
    index = int(TriangleIndices[0])
    spot1 = int(Triangles[index, 0])
    spot2 = int(Triangles[index, 1])
    spot3 = int(Triangles[index, 2])
    spots = np.array([spot1, spot2, spot3])

    indexB = int(TriangleIndices[1])
    spot1B = int(Triangles[indexB, 0])
    spot2B = int(Triangles[indexB, 1])
    spot3B = int(Triangles[indexB, 2])
    spotsB = np.array([spot1B, spot2B, spot3B])

    #making sure swapped edge doesn't already exist
    newEdgeSpots = np.zeros(2)
    oldEdgeSpots = np.zeros(2)
    edgeInd = -1

    for k in range(3):
        for t in range(3):
            if spots[k] == spotsB[t] and edgeInd < 1:
                edgeInd += 1
                oldEdgeSpots[edgeInd] = spots[k]

    for k in range(3):
        if spots[k] != oldEdgeSpots[0] and spots[k] != oldEdgeSpots[1]:
            newEdgeSpots[0] = spots[k]
        if spotsB[k] != oldEdgeSpots[0] and spotsB[k] != oldEdgeSpots[1]:
            newEdgeSpots[1] = spotsB[k]

    eExist = edgeExists(newEdgeSpots, Triangles)

    if not eExist:

        angles = computeAngles(spots, Pts)
        anglesB = computeAngles(spotsB, Pts)

        for q in range(3):
            angleVec[q] = angles[q]
            angleVec[q+3] = anglesB[q]

        angleVec = np.sort(angleVec)

        # creating angle vector with swapped edge
        angleVecSwapped = np.zeros(6)

        spots2 = np.array([newEdgeSpots[0], newEdgeSpots[1], oldEdgeSpots[0]])
        anglesFlipped = computeAngles(spots2, Pts)

        spots2B = np.array([newEdgeSpots[0], newEdgeSpots[1], oldEdgeSpots[1]])
        anglesFlippedB = computeAngles(spots2B, Pts)

        for q in range(3):
            angleVecSwapped[q] = anglesFlipped[q]
            angleVecSwapped[q+3] = anglesFlippedB[q]

        angleVecSwapped = np.sort(angleVecSwapped)

        #swap edge if illegal
        if angleVecSwapped[0] > angleVec[0]:
            print('Flipping edge!!')

            Triangles[index, 0] = newEdgeSpots[0]
            Triangles[index, 1] = newEdgeSpots[1]
            Triangles[index, 2] = oldEdgeSpots[0]

            Triangles[indexB, 0] = newEdgeSpots[0]
            Triangles[indexB, 1] = newEdgeSpots[1]
            Triangles[indexB, 2] = oldEdgeSpots[1]

            #legalize the other edges
            outerEdges = np.zeros((4, 2))
            outerInd = -1
            for w in range(3):
                for z in range(1, 3):
                    if w != z:
                        if Triangles[index, w] != newEdgeSpots[0] and Triangles[index, w] != newEdgeSpots[1]:
                            if outerInd < 1:
                                outerInd += 1
                                outerEdges[outerInd, 0] = Triangles[index, w]
                                outerEdges[outerInd, 1] = Triangles[index, z]

                        elif Triangles[index, z] != newEdgeSpots[0] and Triangles[index, z] != newEdgeSpots[1]:
                            if outerInd < 1:
                                outerInd += 1
                                outerEdges[outerInd, 0] = Triangles[index, w]
                                outerEdges[outerInd, 1] = Triangles[index, z]

            for w in range(3):
                for z in range(1, 3):
                    if w != z:
                        if Triangles[indexB, w] != newEdgeSpots[0] and Triangles[indexB, w] != newEdgeSpots[1]:
                            if outerInd < 3:
                                outerInd += 1
                                outerEdges[outerInd, 0] = Triangles[indexB, w]
                                outerEdges[outerInd, 1] = Triangles[indexB, z]

                        elif Triangles[indexB, z] != newEdgeSpots[0] and Triangles[indexB, z] != newEdgeSpots[1]:
                            if outerInd < 3:
                                outerInd += 1
                                outerEdges[outerInd, 0] = Triangles[indexB, w]
                                outerEdges[outerInd, 1] = Triangles[indexB, z]

            #legalize outer edges
            testEdge = np.array([0, 0])
            for w in range(4):
                testEdge[0] = outerEdges[w, 0]
                testEdge[0] = outerEdges[w, 1]
                legalizeEdge(testEdge, Pts, Triangles)

def CommonEdge(Triangles, indexA, indexB):
    common = False
    edge = np.array([0, 0])
    matchInd = -1

    #check if triangles have common edge
    for m in range(3):
        if Triangles[indexA, m] == Triangles[indexB, 0] or Triangles[indexA, m] == Triangles[indexB, 1] or Triangles[indexA, m] == Triangles[indexB, 2]:
            if matchInd < 1:
                matchInd += 1
                edge[matchInd] = Triangles[indexA, m]
                if matchInd == 1:
                    common = True

    return common, edge


def DelaunayTriangulation(Pts, Triangles):
    newTriangles = Triangles
    for a in range(newTriangles.shape[0]):
        for b in range(newTriangles.shape[0]):
            common, edge = CommonEdge(newTriangles, a, b)
            if common:
                legalizeEdge(edge, Pts, newTriangles)

    return newTriangles

def TriangleSegments(Pts, Triangles):
    TriangleSegs = np.zeros((Triangles.shape[0] * 2, 3))
    steps = 0
    while steps < TriangleSegs.shape[0]:
        index = int(steps / 2)

        spot1 = int(Triangles[index, 0])
        spot2 = int(Triangles[index, 1])
        spot3 = int(Triangles[index, 2])

        TriangleSegs[steps, 0] = Pts[spot1, 1]
        TriangleSegs[steps, 1] = Pts[spot2, 1]
        TriangleSegs[steps, 2] = Pts[spot3, 1]

        TriangleSegs[steps + 1, 0] = Pts[spot1, 2]
        TriangleSegs[steps + 1, 1] = Pts[spot2, 2]
        TriangleSegs[steps + 1, 2] = Pts[spot3, 2]

        steps = steps + 2

    return TriangleSegs

file = "inputTriangulation.txt"
Pts, Triangles = readData(file)
TriangleSegs = TriangleSegments(Pts, Triangles)

newTriangles = Triangles
newTriangles = DelaunayTriangulation(Pts, newTriangles)

newTriangleSegs = TriangleSegments(Pts, newTriangles)

plt.figure()

for i in range(Triangles.shape[0]):
    step = i * 2
    x = np.array([TriangleSegs[step, 0], TriangleSegs[step, 1], TriangleSegs[step, 2], TriangleSegs[step, 0]])
    y = np.array([TriangleSegs[step+1, 0], TriangleSegs[step+1, 1], TriangleSegs[step+1, 2], TriangleSegs[step+1, 0]])
    plt.plot(x, y, 'r-')
for i in range(newTriangles.shape[0]):
    step = i * 2
    x = np.array([newTriangleSegs[step, 0], newTriangleSegs[step, 1], newTriangleSegs[step, 2], newTriangleSegs[step, 0]])
    y = np.array([newTriangleSegs[step+1, 0], newTriangleSegs[step+1, 1], newTriangleSegs[step+1, 2], newTriangleSegs[step+1, 0]])
    plt.plot(x, y, 'b-')
plt.plot(Pts[:, 1], Pts[:, 2], 'k*')
plt.grid()
plt.show()

