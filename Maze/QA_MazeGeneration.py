import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SASampler

# 倒す棒の本数(H × W)
H = 9
W = 9
N = 3 * W + 1

QUBO = {} # Q_h,w,d   (h, w)における棒を倒す方向d


l1 = 1 # コスト(倒れる方向が重ならないように制限)
for i in range(H):
  for j in range(W):
    # 縦方向
    if j == 0:
      if i != H-1:
        QUBO[(2 + 4*j + N*i, 0 + 4*j + N*(i+1))] = QUBO.get((2 + 4*j + N*i, 0 + 4*j + N*(i+1)), 0) + l1
      if i != 0:
        QUBO[(0 + 4*j + N*i, 2 + 4*j + N*(i-1))] = QUBO.get((0 + 4*j + N*i, 2 + 4*j + N*(i-1)), 0) + l1
    else:
      if i != H-1:
        QUBO[(2 + 4+3*(j-1) + N*i, 0 + 4+3*(j-1) + N*(i+1))] = QUBO.get((2 + 4+3*(j-1) + N*i, 0 + 4+3*(j-1) + N*(i+1)), 0) + l1
      if i != 0:
        QUBO[(0 + 4+3*(j-1) + N*i, 2 + 4+3*(j-1) + N*(i-1))] = QUBO.get((0 + 4+3*(j-1) + N*i, 2 + 4+3*(j-1) + N*(i-1)), 0) + l1


l2 = 2 # 罰金法コスト(倒れる方向をひとつに制限)
for i in range(H):
  for j in range(W):
    if j == 0:
      for d1 in range(4):
        for d2 in range(4):
          # 罰金法(倒れる方向をひとつに制限)
          QUBO[(d1 + 4*j + N*i, d2 + 4*j + N*i)] = QUBO.get((d1 + 4*j + N*i, d2 + 4*j + N*i), 0) + l2
        QUBO[(d1 + 4*j + N*i, d1 + 4*j + N*i)] = QUBO.get((d1 + 4*j + N*i, d1 + 4*j + N*i), 0) - 2 * l2
    else:
      for d1 in range(3):
        for d2 in range(3):
          # 罰金法(倒れる方向をひとつに制限)
          QUBO[(d1 + 4+3*(j-1) + N*i, d2 + 4+3*(j-1) + N*i)] = QUBO.get((d1 + 4+3*(j-1) + N*i, d2 + 4+3*(j-1) + N*i), 0) + l2
        QUBO[(d1 + 4+3*(j-1) + N*i, d1 + 4+3*(j-1) + N*i)] = QUBO.get((d1 + 4+3*(j-1) + N*i, d1 + 4+3*(j-1) + N*i), 0) - 2 * l2

'''
# QuantumAnnealer
token = 'XXXX'  # 個人のAPI tokenを使用
endpoint = 'https://cloud.dwavesys.com/sapi/'

dw_sampler = DWaveSampler(solver='Advantage_system6.2', token=token, endpoint=endpoint)
sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(QUBO, num_reads=10)
'''

# SASAmpler
sampler = SASampler()
sampleset = sampler.sample_qubo(QUBO, num_reads=10)

result = [i for i in sampleset.first[0].values()]
field = [[1 for i in range(2 * W + 3)] for j in range(2 * H + 3)]
for i in range(2 * W + 3):
  field[0][i] = 0
  field[-1][i] = 0
for i in range(2 * H + 3):
  field[i][0] = 0
  field[i][-1] = 0
for i in range(2, 2 * H + 3, 2):
  for j in range(2, 2 * W + 3, 2):
    field[i][j] = 0

DIRECTION = [(-1, 0), (0, 1), (1, 0), (0, -1)]
for i in range(H):
  for j in range(W):
    if j == 0:
      for d in range(4):
        if result[d + 4*j + N*i] == 1:
          field[2 * i + 2 + DIRECTION[d][0]][2 * j + 2 + DIRECTION[d][1]] = 0
    else:
      for d in range(3):
        if result[d + 4+3*(j-1) + N*i] == 1:
          field[2 * i + 2 + DIRECTION[d][0]][2 * j + 2 + DIRECTION[d][1]] = 0

plt.imshow(field, cmap='gray')
plt.axis('off')
plt.show()