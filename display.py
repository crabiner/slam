#!/usr/bin/env python3
import cv2
import sdl2.ext

# sdl2 requires init
sdl2.ext.init()

WIDTH = 1920 // 2
HEIGHT = 1080 // 2

window = sdl2.ext.Window("SLAM window", size=(WIDTH, HEIGHT), position=(0, 0))
window.show()

def process_frame(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_KEYDOWN:
            exit(0)
        if event.type == sdl2.SDL_QUIT:
            exit(0)
    surf = sdl2.ext.pixels3d((window.get_surface()))
    surf[:, :, 0:3] = img.swapaxes(0,1)
    window.refresh()
    print(img.shape)


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break