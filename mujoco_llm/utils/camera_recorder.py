import cv2
import numpy as np
from mujoco.glfw import glfw
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE

class CameraRecorder:
    def __init__(self, window, width, height, filename, fps=30, fourcc_str='mp4v'):
        self.window = window
        self.width = width
        self.height = height
        self.filename = filename
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = cv2.VideoWriter(self.filename, fourcc, fps, (width, height))
        self.is_opened = self.writer.isOpened()
        if not self.is_opened:
            raise RuntimeError(f"Failed to open video file for writing: {self.filename}")

    def capture_frame(self):
        glfw.make_context_current(self.window)
        img = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def write(self, frame):
        if self.is_opened:
            self.writer.write(frame)
            
    def capture_and_write(self):
        """캡처 → 바로 저장까지 한 번에"""
        frame = self.capture_frame()
        self.write(frame)

    def release(self):
        if self.is_opened:
            self.writer.release()
            self.is_opened = False
            print(f"Camera view has been recorded to {self.filename}")