sum_d = len(data[2])
            sum_t = len(sum_m)
            A = np.ones((1, sum_d))
            for i in range(0, sum_l):
                arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k] \
                    = arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k] -\
                      A.T * np.linalg.inv(A * A.T) * \
                      (A * arx[((sum_l + sum_t) + i * sum_d): ((sum_l + sum_t) + i * sum_d) + sum_d, k] - 1)
                for j in range(((sum_l + sum_t) + i * sum_d), ((sum_l + sum_t) + i * sum_d) + sum_d):
                    yu = 0
                    if arx[j][k] < 0:
                        arx[j][k] = 0
                        yu = yu + arx[j][k]
                index = 0
                for j in range(((sum_l + sum_t) + i * sum_d), ((sum_l + sum_t) + i * sum_d) + sum_d):
                    if arx[j][k] != 0:
                        index = index + 1
                yu = yu / index
                for j in range(((sum_l + sum_t) + i * sum_d), ((sum_l + sum_t) + i * sum_d) + sum_d):
                    if arx[j][k] != 0:
                        arx[j][k] = arx[j][k] + yu