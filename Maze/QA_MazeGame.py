import random
from collections import deque
from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SASampler
import pygame
import sys
from pygame.locals import *
import time


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


def place_SG(field):
  H = len(field)
  W = len(field[0])
  ry_s = random.randint(1, H - 1)
  rx_s = random.randint(1, W - 1)

  while field[ry_s][rx_s] == 0:
    ry_s = random.randint(1, H - 1)
    rx_s = random.randint(1, W - 1)

  field[ry_s][rx_s] = 2
  
  DIRECTION = [(-1, 0), (0, 1), (1, 0), (0, -1)]
  visitable = set()
  q = deque()
  q.append((ry_s, rx_s))
  while len(q) > 0:
    y, x = q.popleft()
    for dy, dx in DIRECTION:
      ny = y + dy
      nx = x + dx
      if 1 <= ny and ny <= 2 * H + 1 and 1 <= nx and nx <= 2 * W + 1:
        if field[ny][nx] == 1 and (ny, nx) not in visitable:
          visitable.add((ny, nx))
          q.append((ny, nx))

  ry_g, rx_g = random.sample(visitable, 1)[0]
  if ry_g == ry_s and rx_g == rx_s:
    visitable.remove((ry_g, rx_g))
    ry_g, rx_g = random.sample(visitable, 1)[0]

  field[ry_g][rx_g] = 3

  return field, [ry_s, rx_s], [ry_g, rx_g]




BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
CYAN = (0, 255, 255)

def draw_text(bg, txt, x, y, fnt, col): # 影付き文字の表示
    sur = fnt.render(txt, True, BLACK)
    bg.blit(sur, [x+1, y+2])
    sur = fnt.render(txt, True, col)
    bg.blit(sur, [x, y])

def draw_dungeon(bg, field, position):
  h = len(field)
  w = len(field[0])
  x = position[1]
  y = position[0]
  bg.fill(BLACK)
  for dy in range(-2, 3):
    for dx in range(-2, 3):
      ny = y + dy
      nx = x + dx
      if 0 <= ny and ny < h and 0 <= nx and nx < w:
        if field[ny][nx] == 1:
          pygame.draw.rect(bg, WHITE, (100 * (dx + 2), 100 * (dy + 2), 100, 100))
        elif field[ny][nx] == 2:
          pygame.draw.rect(bg, CYAN, (100 * (dx + 2), 100 * (dy + 2), 100, 100))
        elif field[ny][nx] == 3:
          pygame.draw.rect(bg, RED, (100 * (dx + 2), 100 * (dy + 2), 100, 100))


def move_player(key, field, position):
  y = position[0]
  x = position[1]

  if key[K_UP] == 1:
    if field[y-1][x] != 0:
      position[0] -= 1
  elif key[K_DOWN] == 1:
    if field[y+1][x] != 0:
      position[0] += 1
  elif key[K_LEFT] == 1:
    if field[y][x-1] != 0:
      position[1] -= 1
  elif key[K_RIGHT] == 1:
    if field[y][x+1] != 0:
      position[1] += 1
  
  field[y][x] = 1
  field[position[0]][position[1]] = 2

  return field, position


def make_dungeon(QUBO):
  '''
  # QuantumAnnealer
  token = 'XXXX'  # 個人のAPI tokenを使用
  endpoint = 'https://cloud.dwavesys.com/sapi/'

  dw_sampler = DWaveSampler(solver='Advantage_system6.2', token=token, endpoint=endpoint)
  sampler = EmbeddingComposite(dw_sampler)
  sampleset = sampler.sample_qubo(QUBO, num_reads=10)
  '''

  # SASampler
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

  field, position, goal = place_SG(field)

  return field, position, goal


idx = 0
def main():
  global idx

  pygame.init()
  pygame.display.set_caption("Q_maze")
  screen = pygame.display.set_mode((500, 500))
  clock = pygame.time.Clock()
  font = pygame.font.Font(None, 100)
  start = True

  while True:
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit()

    key = pygame.key.get_pressed()
    if idx == 0:
      if not start:
        time.sleep(0.5)
      field, position, goal = make_dungeon(QUBO)
      idx = 1
    if idx == 1: # プレイヤーの移動
      field, position = move_player(key, field, position)
      draw_dungeon(screen, field, position)
      if position[0] == goal[0] and position[1] == goal[1]:
        idx = 2
    elif idx == 2:
      draw_text(screen, "CLEAR!!", 100, 180, font, RED)
      start = False
      idx = 0

    pygame.display.update()
    clock.tick(7)


if __name__ == '__main__':
    main()