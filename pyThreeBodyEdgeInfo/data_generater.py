import numpy as np

data = np.zeros((16*16*16, 3))

offset = 0
for i in range(0, 16):
    x = (i + 0.5)*2 - 16
    for j in range(0, 16):
        y = (j + 0.5)*2 - 16
        for k in range(0, 16):
            z = (k + 0.5)*2 - 16
            data[offset, 0] = x
            data[offset, 1] = y
            data[offset, 2] = z
            offset += 1

np.savetxt('data_sample_test.txt', data)
