from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SASampler


# チェス盤のサイズ
N = 10

QUBO = {} # (i, j)にクイーンを置く

l1 = 2 # 罰金法コスト(各行のクイーンの個数はちょうど1個)
for i in range(N):
  for j1 in range(N):
    for j2 in range(N):
      QUBO[(N * i + j1, N * i + j2)] = QUBO.get((N * i + j1, N * i + j2), 0) + l1
    QUBO[(N * i + j1, N * i + j1)] = QUBO.get((N * i + j1, N * i + j1), 0) - 2 * l1
        
l2 = 2 # 罰金法コスト(各列のクイーンの個数はちょうど1個)
for j in range(N):
  for i1 in range(N):
    for i2 in range(N):
      QUBO[(N * i1 + j, N * i2 + j)] = QUBO.get((N * i1 + j, N * i2 + j), 0) + l2
    QUBO[(N * i1 + j, N * i1 + j)] = QUBO.get((N * i1 + j, N * i1 + j), 0) - 2 * l2

l3 = 1 # 罰金法コスト(右下がり斜めのクイーンの個数は高々1個)
for d in range(1 - N, N):
  for k1 in range(N - abs(d)):
    i1 = max(0, d) + k1
    j1 = max(0, -d) + k1
    for k2 in range(N - abs(d)):
      i2 = max(0, d) + k2
      j2 = max(0, -d) + k2
      QUBO[(N * i1 + j1, N * i2 + j2)] = QUBO.get((N * i1 + j1, N * i2 + j2), 0) + l3
    QUBO[(N * i1 + j1, N * i2 + j2)] = QUBO.get((N * i1 + j1, N * i2 + j2), 0) - 2 * 0.5 * l3

l4 = 1 # 罰金法コスト(右上がり斜めのクイーンの個数は高々1個)
for d in range(1 - N, N):
  for k1 in range(N - abs(d)):
    i1 = max(0, d) + k1
    j1 = min(N + d, N) - k1 - 1
    for k2 in range(N - abs(d)):
      i2 = max(0, d) + k2
      j2 = min(N + d, N) - k2 - 1
      QUBO[(N * i1 + j1, N * i2 + j2)] = QUBO.get((N * i1 + j1, N * i2 + j2), 0) + l4
    QUBO[(N * i1 + j1, N * i2 + j2)] = QUBO.get((N * i1 + j1, N * i2 + j2), 0) - 2 * 0.5 * l4


'''
# QuantumAnnealer
token = 'XXXX'  # 個人のAPI tokenを使用
endpoint = 'https://cloud.dwavesys.com/sapi/'

dw_sampler = DWaveSampler(solver='Advantage_system6.3', token=token, endpoint=endpoint)
sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(QUBO, num_reads=10)
'''

# SASAmpler
sampler = SASampler()
sampleset = sampler.sample_qubo(QUBO, num_reads=10)

result = [i for i in sampleset.first[0].values()]
ans = [['.'] * N for _ in range(N)]
for i in range(N):
  for j in range(N):
    if result[N * i + j] == 1:
      ans[i][j] = '#'

for i in range(N):
  print(*ans[i], sep=' ')