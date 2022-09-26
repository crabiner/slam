#!/usr/bin/env python3
import cv2
import sdl2.ext

# sdl2 requires init
sdl2.ext.init()

WIDTH = 1920 // 2
HEIGHT = 1080 // 2


class Display (object):
    def __init__(self, W, H):
        self.window = sdl2.ext.Window("SLAM window", size=(WIDTH, HEIGHT), position=(0, 0))
        self.window.show()

    def paint(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_KEYDOWN:
                exit(0)
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d((self.window.get_surface()))
        surf[:, :, 0:3] = img.swapaxes(0,1)
        self.window.refresh()

