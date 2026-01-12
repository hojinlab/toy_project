import os
import sys
import time
import threading
from queue import Queue

import mujoco as mj
import cv2
import numpy as np

# 프로젝트 루트에서 utils 가져오기
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.mujoco_renderer import MuJoCoViewer
from utils.object_detector import ObjectDetector

WHEEL_ACTION = {
    "멈춤": (0.0, 0.0),
    "직진": (8.0, 8.0),
    "후진": (-8.0, -8.0),
    "좌회전": (6.0, 8.0),
    "우회전": (8.0, 6.0),
    "제자리 회전": (4.0, -4.0),
}

ARM_ACTIONS = {
    "잡기",
    "놓기",
}

class TurtlebotFactorySim:
    """
    MuJoCo 기반 터틀봇3 팩토리 시뮬 통합 클래스.

    기능:
    - tb3_factory_cards.xml 로드
    - 메인뷰 + 로봇 카메라 렌더링
    - latest_frame 에 로봇 카메라 마지막 프레임(BGR) 저장
    - (옵션) YOLO로 로봇 카메라 프레임 감지 & cv2 창으로 출력
    - (옵션) command_queue 에서 명령을 읽어와 apply_command()로 처리
    """

    def __init__(
        self,
        xml_path: str | None = None,
        use_yolo: bool = False,
        yolo_weight_path: str | None = None,
        yolo_conf: float = 0.5,
        command_queue: Queue | None = None,
        fps: int = 60,
        current_action = None,
        action_end_sim_time = 0.0,
    ):
        # ==== 행동 중 명령 금지위한 초기값 ====
        self.is_busy = False

        # ===== 경로 설정 =====
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(scripts_dir)  # /data/jinsup/js_mujoco

        if xml_path is None:
            xml_path = os.path.join(
                project_root,
                "asset",
                "robotis_tb3",
                "tb3_factory_cards.xml",
            )

        print(f"[TurtlebotFactorySim] Loading scene from: {xml_path}")

        # 검색 모드 타겟 레이블
        self.search_target_label = None  
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_time
        # ===== MuJoCo 모델/데이터 로드 =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # === 센서값 초기화 ===
        self.us_sid, self.us_adr, self.us_dim = self._cache_sensor("ultrasonic")

        # 기존 MuJoCoViewer 사용
        self.viewer = MuJoCoViewer(self.model, self.data)

        # ===== 카메라 프레임 저장용 =====
        # 항상 "로봇 카메라 기준 BGR 이미지"를 최신 상태로 보관
        self.latest_frame: np.ndarray | None = None

        # ===== YOLO 옵션 =====
        self.use_yolo = use_yolo
        self.detector = None
        self.yolo_window_name = "Robot YOLO View"

        if self.use_yolo:
            if yolo_weight_path is None:
                raise ValueError("use_yolo=True 인데 yolo_weight_path 가 없습니다.")
            if not os.path.exists(yolo_weight_path):
                raise FileNotFoundError(f"YOLO weight not found: {yolo_weight_path}")

            print(f"[TurtlebotFactorySim] Loading ObjectDetector: {yolo_weight_path}")
            self.detector = ObjectDetector(yolo_weight_path, conf=yolo_conf)

            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.yolo_window_name, 640, 480)

        # ===== 명령 큐 (LLM / 키보드 등에서 넣어주는 명령) =====
        self.command_queue = command_queue if command_queue is not None else Queue()

        # ===== 루프 설정 =====
        self.fps = fps
        self._running = False

    # ------------------------------------------------------------------
    # 외부에서 사용할 수 있는 유틸 메서드들
    # ------------------------------------------------------------------
    def step_simulation(self):
        """한 타임스텝(fps 기준)만큼 시뮬레이션을 진행."""
        time_prev = self.data.time
        dt = 1.0 / self.fps
        while self.data.time - time_prev < dt:
            self.viewer.step_simulation()

    def render(self):
        """메인뷰 + 로봇 카메라 렌더링, latest_frame 업데이트."""
        # 메인 뷰: IMU overlay
        self.viewer.render_main(overlay_type="imu")

        # 로봇 카메라 화면 표시 + 이미지 캡처
        self.viewer.render_robot()
        # MuJoCoViewer 안에 capture_img() 가 로봇 카메라 뷰를 BGR로 반환한다고 가정
        if hasattr(self.viewer, "capture_img"):
            frame_bgr = self.viewer.capture_img()
            self.latest_frame = frame_bgr
        else:
            self.latest_frame = None

        self.viewer.poll_events()

    # 센서 값 읽어오기
    def _cache_sensor(self, sensor_name: str):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None, None, None
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return sid, adr, dim

    def read_ultrasonic(self):
        if self.us_adr is None:
            return None
        return float(self.data.sensordata[self.us_adr])

    # 명령 처리 로직
    def apply_command(self, cmd: str, base_duration: float = 1.0):
        cmd = cmd.strip()

        # 1) 검색 계열 액션 처리 
        SEARCH_MAP = {
            "SEARCH_HEART": "heart",
            "SEARCH_STAR": "star",   
            "SEARCH_CUBE": "cube",
            "SEARCH_TETRAHEDRON": "tetrahedron",
            "SEARCH_SPHERE": "sphere",
        }

        if self.is_busy:
            print(f"[BUSY] Ignored command: {cmd}")
            return
    
        if cmd in SEARCH_MAP:
            target = SEARCH_MAP[cmd]
            self.search_target_label = target

            self.data.ctrl[0] = 4.0
            self.data.ctrl[1] = -4.0

            self.current_action = cmd
            self.action_end_sim_time = float("inf")

            self.is_busy = True

            print(f"[SEARCH] Start search for '{target}'")
            return

        # 2) 알 수 없는 명령 체크 (버그 수정)
        if cmd not in WHEEL_ACTION and cmd not in ARM_ACTIONS:
            print(f"[TurtlebotFactorySim] Unknown command: {cmd}")
            return

        # 3) ARM 액션 처리 (추가)
        if cmd in ARM_ACTIONS:
            self.apply_arm_action(cmd)
            return

        # 4) 기존 WHEEL 액션 처리 (그대로 유지)
        duration = base_duration
        if cmd in ["좌회전", "우회전"]:
            duration *= 1.6
        elif cmd == "제자리 회전":
            duration *= 1.0

        left, right = WHEEL_ACTION[cmd]
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

        self.current_action = cmd
        self.action_end_sim_time = self.data.time + duration

        self.is_busy = True 

        print(f"[WHEEL] '{cmd}' → L={left}, R={right}, duration={duration:.2f}s")

    def apply_arm_action(self, arm_cmd: str):
        """
        의미 명령을 받아
        어떤 저수준 팔 시퀀스를 실행할지 연결만 한다
        """
        if self.is_busy: #동작중이면 무시
            return
        self.is_busy = True

        if arm_cmd == "잡기":
            self._arm_grasp()
        elif arm_cmd == "놓기":
            self._arm_release()
        else:
            print(f"[ARM] Unknown arm_cmd: {arm_cmd}")
        self.is_busy = False

    # 잡기
    def _arm_grasp(self):
        print("[ARM] Approaching with wheels...")

        # 1️⃣ 바퀴로 전진
        self.data.ctrl[0] = 3.0
        self.data.ctrl[1] = 3.0

        hold_start = None

        while True:
            us = self.read_ultrasonic()
            if us is None:
                continue

            # 2️⃣ 초음파 조건 체크
            if us <= self.ultra_threshold_m:
                if hold_start is None:
                    hold_start = time.time()

                held = time.time() - hold_start
                if held >= self.ultra_hold_sec:
                    print(f"[ARM] Ultrasonic OK (us={us:.3f}m)")
                    break
            else:
                hold_start = None

            time.sleep(0.01)  # 시뮬 프리즈 방지

        # 3️⃣ 바퀴 정지
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

        print("[ARM] Stop & start arm sequence")

        # 4️⃣ 팔 전진
        self.data.ctrl[3] = 0.2
        self.data.ctrl[5] = 0.2
        time.sleep(0.4)

        # 5️⃣ 팔 접기
        self.data.ctrl[2] = 1.57
        self.data.ctrl[4] = -1.57
        time.sleep(0.3)

        # 6️⃣ 손가락 닫기
        self.data.ctrl[7] = -2.36
        self.data.ctrl[8] = 2.36
        self.data.ctrl[10] = -2.36
        self.data.ctrl[11] = 2.36
        time.sleep(0.2)

        # 7️⃣ 압력
        self.data.ctrl[6] = 0.01
        self.data.ctrl[9] = 0.01

        self.arm_state = "HOLDING"
        print("[ARM_SEQ] GRASP COMPLETE")


    def _arm_release(self):
        # 바퀴 멈추기
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

        # 압력 풀기
        self.data.ctrl[6] = 0
        self.data.ctrl[9] = 0
        time.sleep(0.5)

        # 손가락 펴기
        self.data.ctrl[7] = 0
        self.data.ctrl[8] = 0
        self.data.ctrl[10] = 0
        self.data.ctrl[11] = 0
        time.sleep(0.5)

        # 팔 접기
        self.data.ctrl[2] = 1.57
        self.data.ctrl[4] = -1.57
        time.sleep(0.5)

        # 팔 후진
        self.data.ctrl[3] = 0
        self.data.ctrl[5] = 0

        self.arm_state = "IDLE"
        print("[ARM_SEQ] RELEASE")


    def _process_commands(self):
        """command_queue 에 쌓인 명령들을 한 번에 처리."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.apply_command(cmd)

    def yolo_detect_dict(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return {}
        return self.detector.detect_dict(self.latest_frame)

    def yolo_detect_image(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return None
        return self.detector.detect_image(self.latest_frame)

    def _run_yolo_on_latest_frame(self):
        if not self.use_yolo or self.detector is None:
            return
        img_bgr = self.yolo_detect_image()
        if img_bgr is None:
            return
        cv2.imshow(self.yolo_window_name, img_bgr)

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        print("[TurtlebotFactorySim] Start simulation loop.")
        try:
            while self._running and not self.viewer.should_close():
                # 1) 명령 처리
                self._process_commands()

                # 2) 시뮬레이션 한 스텝
                self.step_simulation()

                # 3) 렌더 + latest_frame 갱신
                self.render()

                # 3.5) 검색 모드라면: YOLO로 타겟 감시
                if self.search_target_label is not None:
                    det = self.yolo_detect_dict()
                    if self.search_target_label in det:
                        # 타겟 발견 → 정지 + 검색 종료
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        print(f"[TurtlebotFactorySim] Found '{self.search_target_label}' → stop search.")
                        self.search_target_label = None
                        self.current_action = None
                        self.action_end_sim_time = 0.0
                        self.is_busy = False

                # 4) 일반 액션 duration 기반 정지 (검색 모드일 땐 X)
                if (
                    self.current_action 
                    and not (self.current_action.startswith("SEARCH_"))
                    and self.data.time > self.action_end_sim_time
                ):
                    self.data.ctrl[0] = 0.0
                    self.data.ctrl[1] = 0.0
                    print(f"[TurtlebotFactorySim] '{self.current_action}' 완료 → stop.")
                    self.current_action = None
                    self.is_busy = False

                # 5) YOLO 디스플레이
                if self.use_yolo:
                    self._run_yolo_on_latest_frame()

                # 6) q로 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[TurtlebotFactorySim] 'q' 입력으로 종료합니다.")
                    break

        except Exception as e:
            print(f"\n[TurtlebotFactorySim] 시뮬레이션 중 예외 발생: {e}")
        finally:
            self.close()

    def close(self):
        """시뮬레이션 종료 및 리소스 정리."""
        self._running = False
        if self.use_yolo:
            cv2.destroyWindow(self.yolo_window_name)
        self.viewer.terminate()
        print("[TurtlebotFactorySim] Simulation terminated.")