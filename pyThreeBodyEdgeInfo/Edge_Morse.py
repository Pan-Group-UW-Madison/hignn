import pyThreeBodyEdgeInfo.pyBodyEdgeInfo as pybei
import numpy as np


def create_edge(x, three_body_cutoff, Morse_cutoff, domain):
    edgeInfo = pybei.BodyEdgeInfo()
    edgeInfo.setTwoBodyEpsilon(three_body_cutoff)
    edgeInfo.setThreeBodyEpsilon(Morse_cutoff)
    edgeInfo.setPeriodic(True)
    edgeInfo.setDomain(domain)

    edgeInfo.setTargetSites(x)
    edgeInfo.buildThreeBodyEdgeInfo()
    threeBodyEdgeInfo = edgeInfo.getThreeBodyEdgeInfo()
    threeBodyEdgeSelfInfo = edgeInfo.getThreeBodyEdgeSelfInfo()

    edgeInfo.buildTwoBodyEdgeInfo()
    # twoBodyEdgeInfo = edgeInfo.getTwoBodyEdgeInfo()
    morseForce = edgeInfo.getMorseForce()

    edge_attr3 = edgeInfo.getEdgeAttr3()
    edge_attr_self = edgeInfo.getEdgeAttrSelf()
    edge_attr = edgeInfo.getEdgeAttr()

    # edge_attr3 = np.concatenate(
    #     (morseForce[threeBodyEdgeInfo[0, :]], morseForce[threeBodyEdgeInfo[2, :]]), 1)
    # edge_attr_self = morseForce[threeBodyEdgeSelfInfo[1, :], :]
    # edge_attr = morseForce[edge_info[0, :], :]
    return threeBodyEdgeInfo, threeBodyEdgeSelfInfo, edge_attr, edge_attr3, edge_attr_self, morseForce


def create_edge_2body(Nc):
    edge_info = np.zeros((2, Nc * Nc))
    a1 = np.arange(0, Nc, 1)
    a1 = np.repeat(a1, axis=0, repeats=Nc)
    a2 = np.arange(0, Nc, 1).reshape(1, Nc)
    a2 = np.repeat(a2, axis=0, repeats=Nc)
    a2 = a2.reshape(1, Nc * Nc)
    edge_info[0, :] = a1
    edge_info[1, :] = a2
    edge_info = edge_info.astype(int)
    del_ind = np.arange(0, Nc * (Nc + 1), Nc + 1)
    edge_info = np.delete(edge_info, del_ind, axis=1)

    # edge_attr = force[edge_info[0, :], :]
    return edge_info
