import numpy as np
import time
from Edge_Morse import create_edge, create_edge_2body
import pyBodyEdgeInfo as pybei


# #### preset parameters
data = np.loadtxt('data_sample_test.txt', dtype="float32")
Nc = data.shape[0]
three_body_cutoff = 5.0
Morse_cutoff = 5.0
domain = np.array([[-16, -16, -16], [16, 16, 16]], dtype="float32")
kokkos_obj = pybei.KokkosInterface()
# edgeInfo = pybei.BodyEdgeInfo()
# edgeInfo.setTwoBodyEpsilon(three_body_cutoff)
# edgeInfo.setThreeBodyEpsilon(Morse_cutoff)
# edgeInfo.setPeriodic(True)
# edgeInfo.setDomain(domain)

# #### create two_body edge
edge_info = create_edge_2body(Nc)

N = 1
start = time.time()
for i in range(N):
    threeBodyEdgeInfo, threeBodyEdgeSelfInfo, edge_attr, edge_attr3, edge_attr_self, morseForce \
        = create_edge(data, three_body_cutoff, Morse_cutoff, domain)
end = time.time()
print("{:8.6f}".format((end - start) / N) + 's')
print(threeBodyEdgeInfo.shape)
print(threeBodyEdgeSelfInfo.shape)

print(edge_attr.shape)
print(edge_attr3.shape)
print(edge_attr_self.shape)

print(morseForce)
