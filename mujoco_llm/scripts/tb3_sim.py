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
    - tb3_factory_main.xml 로드
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
            current_action=None,
            action_end_sim_time=0.0,

    ):
        # ==== 행동 중 명령 금지위한 초기값 ====
        self.is_busy = False

        # ===== SEARCH/ALIGN 파라미터 =====
        self.ALIGN_TOL_PX = 12  # 중앙 정렬 허용 오차(픽셀)
        self.SEARCH_TURN_SPEED = 4.0  # 못 찾을 때 회전 속도(바퀴 제어값)
        self.ALIGN_TURN_MAX = 6.0  # 정렬 때 최대 회전 제어값
        self.ALIGN_KP = 0.01  # 픽셀 오차 -> 회전 제어로 바꾸는 비례게인

        # ===== ARM 파라미터(현재 _arm_grasp에서 사용) =====
        self.ultra_threshold_m = 0.15  # 초음파 임계값 (m)
        self.ultra_hold_sec = 0.05  # 임계값 이하 유지 시간 (sec)
        self.arm_state = "IDLE"

        # ===== 경로 설정 =====
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

        # 검색 모드 타겟 레이블
        self.search_target_label = None
        self.approach_target_label = None
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_time
        # ===== MuJoCo 모델/데이터 로드 =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # === 센서값 초기화 ===
        self.us_sid, self.us_adr, self.us_dim = self._cache_sensor("ultrasonic_right")
        self._ultra_hold_start_simtime: float | None = None

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

    def is_ultrasonic_close_enough(self) -> bool:
        """
        초음파 거리 조건 판별 함수

        조건:
        - 거리 <= ultra_threshold_m
        - 그 상태가 ultra_hold_sec 이상 유지

        반환:
        - True  : 충분히 가까움 (안정)
        - False : 아직 멀거나 불안정
        """

        us = self.read_ultrasonic()

        # 센서 없거나 값 이상 → 실패
        if us is None:
            self._ultra_hold_start_simtime = None
            print("[ULTRA] us = None")
            return False

        print(f"[ULTRA] distance = {us:.4f} m")

        # 거리 조건 불만족 → 리셋
        if us > self.ultra_threshold_m:
            self._ultra_hold_start_simtime = None
            return False

        # 거리 조건 처음 만족
        if self._ultra_hold_start_simtime is None:
            self._ultra_hold_start_simtime = float(self.data.time)
            return False

        # 유지 시간 계산
        held = float(self.data.time) - float(self._ultra_hold_start_simtime)

        return held >= self.ultra_hold_sec

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

            self.data.ctrl[0] = self.SEARCH_TURN_SPEED
            self.data.ctrl[1] = -self.SEARCH_TURN_SPEED

            self.current_action = cmd
            self.action_end_sim_time = float("inf")

            self.is_busy = True

            print(f"[SEARCH] Start search for '{target}'")
            return

        APPROACH_MAP = {
            "APPROACH_HEART": "heart",
            "APPROACH_STAR": "star",
            "APPROACH_CUBE": "cube",
            "SAPPROACH_TETRAHEDRON": "tetrahedron",
            "APPROACH_SPHERE": "sphere",
        }

        if self.is_busy:
            print(f"[BUSY] Ignored command: {cmd}")
            return

        if cmd in APPROACH_MAP:
            target = APPROACH_MAP[cmd]
            self.approach_target_label = target

            self.data.ctrl[0] = self.SEARCH_TURN_SPEED# - 100 * (self.data.sensordata[9] - self.data.sensordata[10])
            self.data.ctrl[1] = self.SEARCH_TURN_SPEED
            self.current_action = cmd
            # self.action_end_sim_time = float("inf")

            self.is_busy = True

            print("[ARM] Approaching '{target}'")
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
        if self.is_busy:
            return

        # 잡기 / 놓기는 저수준 제어이므로 busy 처리
        self.is_busy = True

        if arm_cmd == "잡기":
            success = self._arm_grasp()

        elif arm_cmd == "놓기":
            self._arm_release()

        self.is_busy = False

    def wait_sim_time(self, seconds: float):
        """지정한 초(seconds)만큼 시뮬레이션을 돌리며 대기합니다."""
        start_time = self.data.time  # 현재 시뮬레이션 내부 시간
        while self.data.time < start_time + seconds:
            self.step_simulation()  # 물리 연산 수행
            self.render()  # 그래픽 렌더링 (화면 갱신)

    # 잡기
    def _arm_grasp(self):
        # 바퀴 정지
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0
        self.wait_sim_time(1.0)

        print("[ARM] Start arm grasp sequence")

        # ===== 기존 팔 제어 시퀀스 그대로 =====
        self.data.ctrl[3] = 0.2
        self.data.ctrl[5] = 0.2
        self.wait_sim_time(1.0)

        self.data.ctrl[2] = 1.57
        self.data.ctrl[4] = -1.57
        self.wait_sim_time(1.0)

        self.data.ctrl[7] = -2.36
        self.data.ctrl[8] = 2.36
        self.data.ctrl[10] = -2.36
        self.data.ctrl[11] = 2.36
        self.wait_sim_time(1.0)

        self.data.ctrl[6] = 0.01
        self.data.ctrl[9] = 0.01

        self.arm_state = "HOLDING"
        print("[ARM] GRASP COMPLETE")

    def _arm_release(self):
        # 바퀴 멈추기
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0
        self.wait_sim_time(1.0)

        # 압력 풀기
        self.data.ctrl[6] = 0
        self.data.ctrl[9] = 0
        self.wait_sim_time(1.0)

        # 손가락 펴기
        self.data.ctrl[7] = 0
        self.data.ctrl[8] = 0
        self.data.ctrl[10] = 0
        self.data.ctrl[11] = 0
        self.wait_sim_time(1.0)

        # 팔 접기
        self.data.ctrl[2] = 0
        self.data.ctrl[4] = 0
        self.wait_sim_time(1.0)

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

    def _compute_alignment_error_px(self, bbox):
        if bbox is None or self.latest_frame is None:
            return None

        x1, _, x2, _ = bbox
        h, w = self.latest_frame.shape[:2]

        x1 = max(0.0, min(float(w), float(x1)))
        x2 = max(0.0, min(float(w), float(x2)))

        left_margin = float(x1)
        right_margin = float(w) - float(x2)

        # 목표: 0 (정중앙)
        return float(left_margin - right_margin)

    def _get_target_best_bbox(self, det: dict, target_label: str):

        if not det or (target_label not in det):
            return None

        items = det.get(target_label)
        if items is None:
            return None

        # items가 단일 dict일 수도, list일 수도 있음
        if isinstance(items, dict):
            items = [items]

        # 후보 bbox들을 (bbox, score) 형태로 모아서 최고점 선택
        candidates = []

        for it in items:
            # 1) dict 형태: {"bbox":[x1,y1,x2,y2], "conf":0.8} 또는 {"xyxy":[...], "confidence":...}
            if isinstance(it, dict):
                bbox = None
                for key in ("bbox", "xyxy", "box"):
                    if key in it:
                        bbox = it[key]
                        break

                conf = None
                for key in ("conf", "confidence", "score"):
                    if key in it:
                        conf = it[key]
                        break

                if bbox is None:
                    continue

                try:
                    x1, y1, x2, y2 = map(float, bbox[:4])
                except Exception:
                    continue

                # 점수: conf가 있으면 conf, 없으면 bbox 면적
                score = float(conf) if conf is not None else max(0.0, (x2 - x1) * (y2 - y1))
                candidates.append(((x1, y1, x2, y2), score))
                continue

            # 2) list/tuple 형태: [x1,y1,x2,y2,conf] 또는 [x1,y1,x2,y2]
            if isinstance(it, (list, tuple)) and len(it) >= 4:
                try:
                    x1, y1, x2, y2 = map(float, it[:4])
                except Exception:
                    continue

                conf = None
                if len(it) >= 5:
                    try:
                        conf = float(it[4])
                    except Exception:
                        conf = None

                score = conf if conf is not None else max(0.0, (x2 - x1) * (y2 - y1))
                candidates.append(((x1, y1, x2, y2), score))
                continue

        if not candidates:
            return None

        # score 최고인 bbox 반환
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _compute_auto_turn_from_error(self, err_px: float) -> float:

        if err_px is None:
            return 0.0

        kp = 0.015  # 비례 제어 상수
        min_turn = 1.0  # 느릿느릿해지는 것을 방지하는 최소 회전 속도
        max_turn = 4.0  # 최대 회전 속도 제한

        # 1. 정렬 범위 안에 들어왔다면 정지
        if abs(err_px) < self.ALIGN_TOL_PX:
            return 0.0

        # 2. 비례 제어 계산
        turn = err_px * kp

        # 3. 최소/최대 속도 보정 (Deadband 및 Saturation 처리)
        direction = 1 if turn > 0 else -1
        # 최소 속도(min_turn)보다는 크고, 최대 속도(max_turn)보다는 작게 클리핑
        turn = direction * max(min_turn, min(max_turn, abs(turn)))

        return float(turn)

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

                    bbox = self._get_target_best_bbox(det, self.search_target_label)

                    err_px = self._compute_alignment_error_px(bbox)

                    if err_px is None:
                        # 아직 못 찾음 → 계속 회전
                        self.data.ctrl[0] = self.SEARCH_TURN_SPEED
                        self.data.ctrl[1] = -self.SEARCH_TURN_SPEED
                    else:
                        turn_cmd = self._compute_auto_turn_from_error(err_px)

                        # 제자리 회전으로 정렬
                        self.data.ctrl[0] = turn_cmd
                        self.data.ctrl[1] = -turn_cmd

                        # 중앙 정렬 완료 조건
                        if abs(err_px) < self.ALIGN_TOL_PX:
                            self.data.ctrl[0] = 0.0
                            self.data.ctrl[1] = 0.0

                            print("[SEARCH] aligned → stop search")
                            # self.approach_target_label = self.search_target_label
                            self.search_target_label = None
                            self.current_action = None
                            self.is_busy = False

                if self.approach_target_label is not None:
                    if self.is_ultrasonic_close_enough():
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        self.approach_target_label = None
                        self.current_action = None
                        print("초음파 조건 만족")
                        self.is_busy = False
                    else:
                        print("아직 멀어요")
                        self.data.ctrl[0] = self.SEARCH_TURN_SPEED - 50 * (self.data.sensordata[9] - self.data.sensordata[10])
                        self.data.ctrl[1] = self.SEARCH_TURN_SPEED

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
                    # return

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
