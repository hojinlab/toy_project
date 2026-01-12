# utils/keyboard_callbacks.py
import mujoco as mj
from mujoco.glfw import glfw

class KeyboardCallbacks:
    def __init__(self):
        # 라이다 켜진 상태로 시작
        self.lidar_on = True

    def keyboardGLFW(self, window, key, scancode, act, mods, model, data, opt):
        if act == glfw.PRESS or act == glfw.REPEAT:
            if key == glfw.KEY_W:
                data.ctrl[0] = 10
                data.ctrl[1] = 10
            elif key == glfw.KEY_S:
                data.ctrl[0] = 0
                data.ctrl[1] = 0
            elif key == glfw.KEY_X:
                data.ctrl[0] = -10
                data.ctrl[1] = -10
            elif key == glfw.KEY_D:
                data.ctrl[0] = 10
                data.ctrl[1] = -10
            elif key == glfw.KEY_A:
                data.ctrl[0] = -10
                data.ctrl[1] = 10
            elif key == glfw.KEY_BACKSPACE:
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)
            elif key == glfw.KEY_O:
                data.ctrl[3] = 0.2
                data.ctrl[5] = 0.2
            elif key == glfw.KEY_P:
                data.ctrl[3] = 0
                data.ctrl[5] = 0
            elif key == glfw.KEY_U:
                data.ctrl[2] = 1.57
                data.ctrl[4] = -1.57
            elif key == glfw.KEY_I:
                data.ctrl[2] = 0
                data.ctrl[4] = 0
            elif key == glfw.KEY_N:
                data.ctrl[6] = 0.01
                data.ctrl[9] = 0.01
            elif key == glfw.KEY_M:
                data.ctrl[6] = 0
                data.ctrl[9] = 0
            elif key == glfw.KEY_V:
                data.ctrl[7] = -2.36
                data.ctrl[8] = 2.36
                data.ctrl[10] = -2.36
                data.ctrl[11] = 2.36
            elif key == glfw.KEY_B:
                data.ctrl[7] = 0
                data.ctrl[8] = 0
                data.ctrl[10] = 0
                data.ctrl[11] = 0

            # L 키로 라이다 빔 시각화 토글
            elif key == glfw.KEY_L:
                flag = mj.mjtVisFlag.mjVIS_RANGEFINDER
                self.lidar_on = not self.lidar_on
                opt.flags[flag] = 1 if self.lidar_on else 0

        if act == glfw.RELEASE:
            if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D, glfw.KEY_X]:
                data.ctrl[0] = 0
                data.ctrl[1] = 0
