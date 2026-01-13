import threading
import json
import yaml
import os
import re
from queue import Queue
from dotenv import load_dotenv
import ollama  # Google GenAI 대신 ollama 사용

# YOLO
from ultralytics import YOLO
import cv2

load_dotenv()

TARGET_MAP = {
    # 정사면체 (Tetrahedron)
    "정사면체": "tetrahedron",
    "사면체": "tetrahedron",
    "tetrahedron": "tetrahedron",
    "tetra": "tetrahedron",
    "삼각뿔": "tetrahedron",
    "세모": "tetrahedron",

    # 정육면체 (Cube)
    "정육면체": "cube",
    "육면체": "cube",
    "정6면체": "cube",
    "큐브": "cube",
    "cube": "cube",
    "상자": "cube",
    "box": "cube",
    "네모": "cube",

    # 구 (Sphere)
    "구": "sphere",
    "구체": "sphere",
    "공": "sphere",
    "sphere": "sphere",
    "ball": "sphere",
    "둥근것": "sphere",
    "동그라미": "sphere",

    # 별 (Star)
    "별": "star",
    "별모양": "star",
    "스타": "star",
    "star": "star",
    "오각형별": "star",

    # 하트 (Heart)
    "하트": "heart",
    "심장": "heart",
    "하트모양": "heart",
    "heart": "heart",
    "사랑": "heart",
}

SEARCH_CMD = {
    "tetrahedron": "SEARCH_TETRAHEDRON",
    "cube": "SEARCH_CUBE",
    "sphere": "SEARCH_SPHERE",
    "star": "SEARCH_STAR",
    "heart": "SEARCH_HEART",
}


# ============================================
# QWEN LLM RUNNER FOR TURTLEBOT3 (Ollama)
# ============================================

class QwenTb3:
    def __init__(self, prompt_path, model="gemma2:9b", command_queue=None):
        self.command_queue = command_queue if command_queue else Queue()

        # Load prompt.yaml
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_instruction = yaml.safe_load(f)["template"]

        self.model_name = model

        # threads
        self.thread = None
        self.stop_event = threading.Event()

    # ----------------------------------------
    def run_qwen(self, question, detection_json):
        """Qwen2.5 (Ollama)에게 분석 요청"""
        print(f"[QwenTb3] Using model: {self.model_name}")

        user_content = f"""
# 감지된 객체 정보(JSON):
{detection_json}

# 질문:
{question}
"""
        try:
            # Ollama API call
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_instruction},
                    {'role': 'user', 'content': user_content},
                ],
                options={
                    'temperature': 0.1
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Qwen Error: {e}"

    # ----------------------------------------
    def _extract_target_from_question(self, q: str) -> str | None:
        q_low = q.lower()
        keys = sorted(TARGET_MAP.keys(), key=len, reverse=True)
        for k in keys:
            if k.lower() in q_low:
                return TARGET_MAP[k]
        return None

    # ----------------------------------------
    def talk(self, sim):
        was_busy = False

        while not self.stop_event.is_set():
            if was_busy and (not sim.is_busy):
                print("✅ 동작이 끝났습니다. 다음 명령을 입력하세요.")

            try:
                question = input("\n💬 Human: ")

                # YOLO detection
                det_dict = sim.yolo_detect_dict() or {}
                det_json = json.dumps(det_dict, ensure_ascii=False, indent=2)

                # 목표 카드 추출
                target = self._extract_target_from_question(question)

                # 1) 목표가 있는데 화면에 없으면: SEARCH 모드로 전환
                if target and target not in det_dict:
                    cmd = SEARCH_CMD[target]
                    print(f"➡️ '{target}'가 안보여서 {cmd}로 탐색할게요.")
                    self.command_queue.put(cmd)
                    continue

                # 2) Qwen 호출
                answer = self.run_qwen(question, det_json)
                print(f"\n🤖 Qwen:\n{answer}\n")

                # 3) Action 추출
                action_match = re.search(r"Action:\s*([^\n]+)", answer)
                action = action_match.group(1).strip() if action_match else ""

                # 4) Action 실행
                if action:
                    print(f"➡️ Extracted Action: {action}")
                    self.command_queue.put(action)

            except EOFError:
                break

    # ----------------------------------------
    # Qwen + YOLO 스레드 시작
    def start(self, sim):
        self.thread = threading.Thread(target=self.talk, args=(sim,), daemon=True)
        self.thread.start()