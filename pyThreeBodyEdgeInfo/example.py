import pyBodyEdgeInfo as pytbei
import numpy as np
import Edge_construction_transfer as edge_tf
import time

kokkos_obj = pytbei.KokkosInterface()

data = np.loadtxt('data_sample.txt', dtype="float32")
print(data.shape)

edgeInfo = pytbei.BodyEdgeInfo()
edgeInfo.setThreeBodyEpsilon(5.0)
start = time.time()
for i in range(0, 20):
    edgeInfo.setTargetSites(data)
    edge = edgeInfo.getThreeBodyEdgeInfo()
end = time.time()
print("{:8.6f}".format((end - start) / 20) + 's')

force = np.zeros((data.shape[0]), dtype="float32")

start = time.time()
for i in range(0, 20):
    edgeAttr = edgeInfo.getThreeBodyEdgeAttr(force)
end = time.time()
print("{:8.6f}".format((end - start) / 20) + 's')
# print(edgeAttr.shape)


# force = np.zeros((data.shape[0], 3))
# start = time.time()
# for i in range(0, 20):
#     edge_info, edge_attr, edge_info_self, edge_attr_self = edge_tf.create_edge_3body(
#         data[:, 0:3], data.shape[0], 3, 10, force)
# end = time.time()
# print("{:8.6f}".format((end - start) / 20) + 's')

# print(edge_attr_self.shape)

# result = edge_attr_self.astype(int)
# np.savetxt('result.txt', result.T, '%d')
