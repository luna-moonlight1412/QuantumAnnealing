import os
import cv2
import numpy as np

from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SASampler

import pygame
import sys
from pygame.locals import *
import matplotlib.pyplot as plt


POINT_SIZE = 4 # 点の大きさ
LINE_WIDTH = 2 # 線の太さ
SELECT_RADIUS = 10 # 選択の大きさ
NEIGHBOR_RADIUS = 10 # 頂点同士の距離の許容範囲
LINE_COLOR = (255, 0, 0) # 線の色(RGB)
POINT_COLOR = (255, 0, 0) # 点の色(RGB)
SCREEN_COLOR = (255, 255, 255)

COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
COLOR_NUM = len(COLOR)


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

X, Y = 0, 0
click = False
mode = 'vertex'

vertex = []
edges = []
selected_point = None



# 写真データ
path = './image' #ディレクトリ名
PICTURES = os.listdir(path)

img = cv2.imread("image/" + PICTURES[0], cv2.IMREAD_COLOR)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

# 画像の大きさを取得
HEIGHT, WIDTH, c= img.shape[:3]

PICTURE = pygame.image.load("image/" + str(PICTURES[0]))





def cross(v, u):
  return v[0] * u[1] - v[1] * u[0]

def cross_check(edge1, edge2):
  A = ((edge1[1][0] - edge1[0][0]), (edge1[1][1] - edge1[0][1]))
  B = ((edge2[0][0] - edge1[0][0]), (edge2[0][1] - edge1[0][1]))
  C = ((edge2[1][0] - edge1[0][0]), (edge2[1][1] - edge1[0][1]))

  D = ((edge2[1][0] - edge2[0][0]), (edge2[1][1] - edge2[0][1]))
  E = ((edge1[0][0] - edge2[0][0]), (edge1[0][1] - edge2[0][1]))
  F = ((edge1[1][0] - edge2[0][0]), (edge1[1][1] - edge2[0][1]))

  if cross(A, B) * cross(A, C) < 0 and cross(D, E) * cross(D, F) < 0:
    return True
  else:
    return False
  

def change2QUBO(l1=1, l2=1):
  QUBO = {}

  n = len(vertex)

  idx_edges = {}
  for i in range(n):
    idx_edges[i] = []

  for v, u in edges:
    v = vertex.index(v)
    u = vertex.index(u)
    if u not in idx_edges[v]:
      idx_edges[v].append(u)
      idx_edges[u].append(v)

  # 罰金法コスト(選ぶ色をひとつに制限)
  for v in range(n):
    for c1 in range(COLOR_NUM):
      for c2 in range(COLOR_NUM):
        # 罰金法(選ぶ色をひとつに制限)
        QUBO[(c1 + COLOR_NUM*v, c2 + COLOR_NUM*v)] = QUBO.get((c1 * COLOR_NUM*v, c2 + COLOR_NUM*v), 0) + l1
      QUBO[(c1 + COLOR_NUM*v, c1 + COLOR_NUM*v)] = QUBO.get((c1 + COLOR_NUM*v, c1 + COLOR_NUM*v), 0) - 2 * l1

  # コスト(選ぶ色が辺の両端で重ならないように制限)
  for v in range(n):
    for nv in idx_edges[v]:
      for c in range(COLOR_NUM):
        QUBO[(c + COLOR_NUM*v, c + COLOR_NUM*nv)] = QUBO.get((c * COLOR_NUM*v, c + COLOR_NUM*nv), 0) + l2

  return QUBO


def culc_QA(QUBO, solver='openjij', token='', num_reads=10):
  if solver == 'openjij':
    sampler = SASampler()
    sampleset = sampler.sample_qubo(QUBO, num_reads=num_reads)

  elif solver == 'dwave':
    token = token  # 個人のAPI tokenを使用
    endpoint = 'https://cloud.dwavesys.com/sapi/'

    dw_sampler = DWaveSampler(solver='Advantage_system4.1', token=token, endpoint=endpoint)
    sampler = EmbeddingComposite(dw_sampler)
    sampleset = sampler.sample_qubo(QUBO, num_reads=num_reads)

  result = [i for i in sampleset.first[0].values()]

  return result


def coloring(result):
  mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
  for v in range(len(vertex)):
    for c in range(COLOR_NUM):
      if result[c + COLOR_NUM * v] == 1:
        cv2.floodFill(img, 
                      mask=mask,
                      seedPoint=vertex[v],
                      newVal=COLOR[c],
                      loDiff=(25, 25, 25),
                      upDiff=(20, 20, 20),)

def draw_text(bg, txt, x, y, fnt, col): # 影付き文字の表示
    sur = fnt.render(txt, True, BLACK)
    bg.blit(sur, [x + 1, y + 2])
    sur = fnt.render(txt, True, col)
    bg.blit(sur, [x, y])

def draw(bg, fnt, result=False):
  global vertex

  bg.fill(SCREEN_COLOR)
  pygame.draw.rect(bg, BLACK, (0, 0, WIDTH, HEIGHT+1), 2)
  if result:
    result_img = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], 'RGB')
    bg.blit(result_img, (0, 0))
  else:
    bg.blit(PICTURE, (0, 0))

  # 選択肢
  pygame.draw.rect(bg, BLACK, (WIDTH, 0, 50, 50), 2)
  pygame.draw.circle(bg, POINT_COLOR, (WIDTH + 25, 25), 10)

  pygame.draw.rect(bg, BLACK, (WIDTH, 50, 50, 50), 2)
  pygame.draw.line(bg, LINE_COLOR, (WIDTH + 10, 70), (WIDTH + 40, 80), 3)

  pygame.draw.rect(bg, BLACK, (WIDTH, 100, 50, 50), 2)
  pygame.draw.rect(bg, BLACK, (WIDTH + 15, 110, 20, 10), 2)
  pygame.draw.rect(bg, BLUE, (WIDTH + 15, 120, 20, 20))

  pygame.draw.rect(bg, BLACK, (WIDTH, 150, 50, 50), 2)
  pygame.draw.line(bg, LINE_COLOR, (WIDTH + 10, 170), (WIDTH + 20, 173), 3)
  pygame.draw.line(bg, LINE_COLOR, (WIDTH + 30, 177), (WIDTH + 40, 180), 3)
  pygame.draw.line(bg, BLACK, (WIDTH + 20, 180), (WIDTH + 30, 170), 3)

  pygame.draw.rect(bg, GREEN, (WIDTH, max(300, HEIGHT) - 100, 50, 100))
  pygame.draw.polygon(bg, BLACK, [(WIDTH + 10, max(300, HEIGHT) - 60), (WIDTH + 10, max(300, HEIGHT) - 40), (WIDTH + 25, max(300, HEIGHT) - 40), (WIDTH + 25, max(300, HEIGHT) - 30), (WIDTH + 40, max(300, HEIGHT) - 50), (WIDTH + 25, max(300, HEIGHT) - 70), (WIDTH + 25, max(300, HEIGHT) - 60)], 0)

  if result:
    pygame.draw.rect(bg, GREEN, (0, max(300, HEIGHT), 300, 100))
    draw_text(bg, 'SAVE', 50, HEIGHT + 15, fnt, BLACK)

    pygame.draw.rect(bg, RED, (max(550, WIDTH) - 250, max(300, HEIGHT), 300, 100))
    draw_text(bg, 'UNDO', max(550, WIDTH) - 200, HEIGHT + 15, fnt, BLACK)

  # どれを選んだかわかりやすく
  s = pygame.Surface((50, 50))
  s.set_alpha(128)
  s.fill(BLACK)
  if mode == 'vertex':
    bg.blit(s, (WIDTH, 0))
  elif mode == 'edge':
    bg.blit(s, (WIDTH, 50))
  elif mode == 'erase':
    bg.blit(s, (WIDTH, 100))
  elif mode == 'scissors':
    bg.blit(s, (WIDTH, 150))

  # グラフを表示
  if not result:
    for vx, vy in vertex:
      pygame.draw.circle(bg, POINT_COLOR, (vx, vy), POINT_SIZE)
    for v1, v2 in edges:
      pygame.draw.line(bg, LINE_COLOR, v1, v2, LINE_WIDTH)

  # 選択を表示
  if selected_point != None:
    if mode == 'edge':
      for rx, ry in vertex:
        if (rx - X) ** 2 + (ry - Y) ** 2 <= SELECT_RADIUS ** 2:
          pygame.draw.line(bg, LINE_COLOR, (selected_point), (rx, ry), LINE_WIDTH)
          break
      else:
        pygame.draw.line(bg, LINE_COLOR, (selected_point), (X, Y), LINE_WIDTH)

    elif mode == 'scissors':
      pygame.draw.line(bg, BLACK, (selected_point), (X, Y), 3)



def click_screen(result=False):
  global vertex, sleep, mode, selected_point, idx, img

  if result:
    if click:
      if 0 <= X and X <= 300 and HEIGHT <= Y and Y <= HEIGHT + 100:
        cv2.imwrite('results/colored.png', img[:,:,::-1])
      elif max(550, WIDTH) - 250 <= X and X <= max(550, WIDTH) + 50 and HEIGHT <= Y and Y <= HEIGHT + 100:
        img = cv2.imread("image/" + PICTURES[0], cv2.IMREAD_COLOR)
        return True

  else:
    if click and sleep > 5:
      # 画像をクリック
      if 0 <= X and X <= WIDTH and 0 <= Y and Y <= HEIGHT:
        if mode == 'vertex':
          if len(vertex) == 0:
            vertex.append((X, Y))
            sleep = 0
          else:
            for rx, ry in vertex:
              if (rx - X) ** 2 + (ry - Y) ** 2 <= NEIGHBOR_RADIUS ** 2:
                break
            else:
              vertex.append((X, Y))
              sleep = 0

        if mode == 'edge':
          if selected_point == None:
            for rx, ry in vertex:
              if (rx - X) ** 2 + (ry - Y) ** 2 <= SELECT_RADIUS ** 2:
                if selected_point != (X, Y):
                  selected_point = (rx, ry)

        if mode == 'erase':
          for rx, ry in vertex:
            if (rx - X) ** 2 + (ry - Y) ** 2 <= SELECT_RADIUS ** 2:
              vertex.remove((rx, ry))
              delete_edge = []
              for edge in edges:
                if (rx, ry) in edge:
                  delete_edge.append(edge)
              for edge in delete_edge:
                edges.remove(edge)

        if mode == 'scissors':
          if selected_point == None:
            selected_point = (X, Y)
      
      # 画像外をクリック
      else:
        if 0 <= Y and Y <= 50:
          mode = 'vertex'
        elif 50 <= Y and Y <= 100:
          mode = 'edge'
        elif 100 <= Y and Y <= 150:
          mode = 'erase'
        elif 150 <= Y and Y <= 200:
          mode = 'scissors'
        elif max(300, HEIGHT) - 100 <= Y and Y <= max(300, HEIGHT):
          sleep = 0
          idx = 1

    # クリックを離す
    elif not click:
      if mode == 'edge':
        if selected_point != None:
          for rx, ry in vertex:
            if selected_point != (rx, ry):
              if (rx - X) ** 2 + (ry - Y) ** 2 <= SELECT_RADIUS ** 2:
                edges.append((selected_point, (rx, ry)))

          selected_point = None

      if mode == 'scissors':
        if selected_point != None:
          delete_edge = []
          for edge in edges:
            if cross_check(edge, (selected_point, (X, Y))):
              delete_edge.append(edge)

          for edge in delete_edge:
            edges.remove(edge)

        selected_point = None

  return False



idx = 0
sleep = 0

def main():
  global idx, X, Y, click, sleep

  pygame.init()
  pygame.display.set_caption("Q_coloring")
  screen = pygame.display.set_mode((max(550, WIDTH) + 50, max(300, HEIGHT) + 100))
  clock = pygame.time.Clock()
  font = pygame.font.Font(None, 100)


  while True:
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit()

    key = pygame.key.get_pressed()
    X, Y = pygame.mouse.get_pos()
    click, Btn2, Btn3 = pygame.mouse.get_pressed()

    if idx == 0:
      draw(screen, font)
      click_screen()

      sleep += 1

    elif idx == 1:
      QUBO = change2QUBO(l1=1, l2=1)
      result = culc_QA(QUBO, solver='openjij', num_reads=10)
      coloring(result)
      idx = 2
    
    elif idx == 2:
      draw(screen, font, result=True)
      if click_screen(result=True):
        idx = 0


    pygame.display.update()
    clock.tick(30)


if __name__ == '__main__':
    main()