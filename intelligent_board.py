import pygame as pg
import numpy as np
from matplotlib.image import imsave
from joblib import load
import sys
import os

AI_IMG: int = 28
WIDTH: int = AI_IMG * 18
HEIGHT: int = AI_IMG * 18
FPS: int = 60

pg.init()

pg.display.set_caption("Intelligent Board")
display = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()

model = load("knn_model.joblib")
board = np.zeros((AI_IMG, AI_IMG))
font = pg.font.Font("./Minecraft.ttf", 32)
prediction = None
result_text = font.render("The written digit is: " + str(prediction), False, (255, 255, 255))


while True:
    
    for ev in pg.event.get():
        if ev.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if ev.type == pg.KEYDOWN:
            if ev.key == pg.K_SPACE:
                board = np.zeros((AI_IMG, AI_IMG))
                result_text = font.render("The written digit is: " + str(None), False, (255, 255, 255))
            if ev.key == pg.K_RETURN:
                processed_img = np.fliplr(np.rot90(board / 255.0, -1))
                prediction = model.predict(processed_img.reshape(1, -1))
                result_text = font.render("The written digit is: " + str(prediction[0]), False, (255, 255, 255))
                imsave("temp.png", processed_img, cmap="gray")

    mouse = pg.mouse.get_pressed()
    
    if any(mouse):
        x, y = pg.mouse.get_pos()
        x, y = int((x / WIDTH) * AI_IMG), int((y / HEIGHT) * AI_IMG)
        if mouse[0]:
            board[x, y] = 255
        if mouse[2]:
            board[x, y] = 0
                
    display.fill((255, 255, 255))
    
    rendering_board = np.zeros((AI_IMG, AI_IMG, 3))
    for x in range(AI_IMG):
        for y in range(AI_IMG):
            rendering_board[x, y] = board[x, y], board[x, y], board[x, y]
    surface = pg.surfarray.make_surface(rendering_board)
    scaled_up_surf = pg.transform.scale(surface, (WIDTH, HEIGHT))
    display.blit(scaled_up_surf , (0, 0))
    display.blit(result_text, (0, 0))
            
    pg.display.update()
    clock.tick(FPS)