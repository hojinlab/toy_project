import os
import sys
import time
from queue import Queue

import mujoco as mj
import cv2
import numpy as np

# 프로젝트 루트에서 utils 가져오기
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.mujoco_renderer import MuJoCoViewer
from utils.object_detector import ObjectDetector


# ============================================================
# Action 정의
# ============================================================
WHEEL_ACTION = {
    "멈춤": (0.0, 0.0),
    "직진": (8.0, 8.0),
    "후진": (-8.0, -8.0),
    "좌회전": (6.0, 8.0),
    "우회전": (8.0, 6.0),
    "제자리 회전": (4.0, -4.0),
}

ARM_ACTIONS = {"잡기", "놓기"}


# ============================================================
# TurtlebotFactorySim
# ============================================================
class TurtlebotFactorySim:
    """
    MuJoCo 기반 터틀봇3 팩토리 시뮬 통합 클래스
    """

    def __init__(
        self,
        xml_path=None,
        use_yolo=False,
        yolo_weight_path=None,
        yolo_conf=0.5,
        command_queue=None,
        fps=60,
        current_action=None,
        action_end_sim_time=0.0,
    ):
        # ===== 상태 플래그 =====
        self.is_busy = False

        # ===== SEARCH / ALIGN 파라미터 =====
        self.ALIGN_TOL_PX = 12
        self.SEARCH_TURN_SPEED = 4.0
        self.ALIGN_KP = 0.015

        # ===== ARM / 초음파 =====
        self.ultra_threshold_m = 0.05
        self.ultra_hold_sec = 0.05
        self.arm_state = "IDLE"

        # ===== 경로 =====
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(scripts_dir)

        if xml_path is None:
            xml_path = os.path.join(
                project_root,
                "asset",
                "robotis_tb3",
                "tb3_factory_main.xml",
            )

        print(f"[TurtlebotFactorySim] Loading scene from: {xml_path}")

        # ===== 탐색 타겟 =====
        self.search_target_label = None
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_time

        # ===== MuJoCo =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # ===== 센서 =====
        self.us_sid, self.us_adr, self.us_dim = self._cache_sensor("ultrasonic")

        # ===== Viewer =====
        self.viewer = MuJoCoViewer(self.model, self.data)

        # ===== 카메라 프레임 =====
        self.latest_frame = None

        # ===== YOLO =====
        self.use_yolo = use_yolo
        self.detector = None
        self.yolo_window_name = "Robot YOLO View"

        if self.use_yolo:
            if yolo_weight_path is None:
                raise ValueError("YOLO weight path missing")
            self.detector = ObjectDetector(yolo_weight_path, conf=yolo_conf)
            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)

        self.command_queue = command_queue if command_queue else Queue()
        self.fps = fps
        self._running = False

    # ============================================================
    # 기본 유틸
    # ============================================================
    def _cache_sensor(self, sensor_name):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None, None, None
        return sid, int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def read_ultrasonic(self):
        if self.us_adr is None:
            return None
        return float(self.data.sensordata[self.us_adr])

    def step_simulation(self):
        time_prev = self.data.time
        dt = 1.0 / self.fps
        while self.data.time - time_prev < dt:
            self.viewer.step_simulation()

    def render(self):
        self.viewer.render_main(overlay_type="imu")
        self.viewer.render_robot()
        if hasattr(self.viewer, "capture_img"):
            self.latest_frame = self.viewer.capture_img()
        self.viewer.poll_events()

    # ============================================================
    # 명령 처리
    # ============================================================
    def apply_command(self, cmd, base_duration=1.0):
        cmd = cmd.strip()

        SEARCH_MAP = {
            "SEARCH_HEART": "heart",
            "SEARCH_STAR": "star",
            "SEARCH_CUBE": "cube",
            "SEARCH_TETRAHEDRON": "tetrahedron",
            "SEARCH_SPHERE": "sphere",
        }

        if self.is_busy:
            return

        # --- SEARCH ---
        if cmd in SEARCH_MAP:
            self.search_target_label = SEARCH_MAP[cmd]
            self.current_action = cmd
            self.action_end_sim_time = float("inf")
            self.is_busy = True
            print(f"[SEARCH] Start search: {self.search_target_label}")
            return

        # --- ARM ---
        if cmd in ARM_ACTIONS:
            self.apply_arm_action(cmd)
            return

        # --- WHEEL ---
        if cmd not in WHEEL_ACTION:
            print(f"[WARN] Unknown command: {cmd}")
            return

        duration = base_duration
        if cmd in ["좌회전", "우회전"]:
            duration *= 1.6

        l, r = WHEEL_ACTION[cmd]
        self.data.ctrl[0] = l
        self.data.ctrl[1] = r

        self.current_action = cmd
        self.action_end_sim_time = self.data.time + duration
        self.is_busy = True

        print(f"[WHEEL] {cmd} ({duration:.2f}s)")

    # ============================================================
    # ARM ACTION (MODIFIED)
    # ============================================================
    def apply_arm_action(self, arm_cmd):
        if self.is_busy:
            return

        self.is_busy = True

        if arm_cmd == "잡기":
            success = self._arm_grasp()

            # 🔧 초음파 실패 → 탐색 복귀
            if not success:
                print("[ARM] Grasp failed → back to SEARCH")
                self.is_busy = False
                return

        elif arm_cmd == "놓기":
            self._arm_release()

        self.is_busy = False

    # ============================================================
    # ARM GRASP (MODIFIED)
    # ============================================================
    def _arm_grasp(self):
        print("[ARM] Approaching object")

        self.data.ctrl[0] = 3.0
        self.data.ctrl[1] = 3.0

        hold_start = None
        timeout_start = time.time()
        TIMEOUT = 3.0

        while True:
            if time.time() - timeout_start > TIMEOUT:
                self.data.ctrl[0] = 0.0
                self.data.ctrl[1] = 0.0
                print("[ARM] Ultrasonic timeout")
                return False

            us = self.read_ultrasonic()
            if us is None:
                time.sleep(0.01)
                continue

            if us <= self.ultra_threshold_m:
                if hold_start is None:
                    hold_start = time.time()
                if time.time() - hold_start >= self.ultra_hold_sec:
                    break
            else:
                hold_start = None

            time.sleep(0.01)

        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

        # ---- Arm sequence ----
        self.data.ctrl[3] = 0.2
        self.data.ctrl[5] = 0.2
        time.sleep(0.4)

        self.data.ctrl[2] = 1.57
        self.data.ctrl[4] = -1.57
        time.sleep(0.3)

        self.data.ctrl[7] = -2.36
        self.data.ctrl[8] = 2.36
        self.data.ctrl[10] = -2.36
        self.data.ctrl[11] = 2.36
        time.sleep(0.2)

        self.data.ctrl[6] = 0.01
        self.data.ctrl[9] = 0.01

        self.arm_state = "HOLDING"
        print("[ARM] GRASP COMPLETE")
        return True

    def _arm_release(self):
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

        self.data.ctrl[6] = 0
        self.data.ctrl[9] = 0
        time.sleep(0.3)

        self.data.ctrl[7] = 0
        self.data.ctrl[8] = 0
        self.data.ctrl[10] = 0
        self.data.ctrl[11] = 0
        time.sleep(0.3)

        self.arm_state = "IDLE"
        print("[ARM] RELEASE")

    # ============================================================
    # YOLO & ALIGN
    # ============================================================
    def yolo_detect_dict(self):
        if not self.use_yolo or self.latest_frame is None:
            return {}
        return self.detector.detect_dict(self.latest_frame)

    def _compute_alignment_error_px(self, bbox):
        if bbox is None or self.latest_frame is None:
            return None
        x1, _, x2, _ = bbox
        h, w = self.latest_frame.shape[:2]
        return float(x1 - (w - x2))

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def start(self):
        self._running = True
        print("[SIM] Start")

        try:
            while self._running and not self.viewer.should_close():
                while not self.command_queue.empty():
                    self.apply_command(self.command_queue.get())

                self.step_simulation()
                self.render()

                # --- SEARCH MODE ---
                if self.search_target_label:
                    det = self.yolo_detect_dict()
                    items = det.get(self.search_target_label)

                    if not items:
                        self.data.ctrl[0] = self.SEARCH_TURN_SPEED
                        self.data.ctrl[1] = -self.SEARCH_TURN_SPEED
                    else:
                        bbox = items[0]["bbox"] if isinstance(items, list) else None
                        err = self._compute_alignment_error_px(bbox)

                        if err is None or abs(err) > self.ALIGN_TOL_PX:
                            turn = err * self.ALIGN_KP if err else self.SEARCH_TURN_SPEED
                            self.data.ctrl[0] = turn
                            self.data.ctrl[1] = -turn
                        else:
                            self.data.ctrl[0] = 0.0
                            self.data.ctrl[1] = 0.0
                            print("[SEARCH] Aligned")
                            self.search_target_label = None
                            self.current_action = None
                            self.is_busy = False

                # --- ACTION END ---
                if (
                    self.current_action
                    and not self.current_action.startswith("SEARCH")
                    and self.data.time > self.action_end_sim_time
                ):
                    self.data.ctrl[0] = 0.0
                    self.data.ctrl[1] = 0.0
                    self.current_action = None
                    self.is_busy = False

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.close()

    def close(self):
        self._running = False
        if self.use_yolo:
            cv2.destroyAllWindows()
        self.viewer.terminate()
        print("[SIM] Terminated")
