import threading
import json
import yaml
import os
import re
from queue import Queue
from dotenv import load_dotenv
import ollama  # Ollama 사용

load_dotenv()

# ==================================================
# Target Mapping (자연어 → YOLO label)
# ==================================================
TARGET_MAP = {
    # Tetrahedron
    "정사면체": "tetrahedron",
    "사면체": "tetrahedron",
    "tetrahedron": "tetrahedron",
    "tetra": "tetrahedron",
    "삼각뿔": "tetrahedron",
    "세모": "tetrahedron",

    # Cube
    "정육면체": "cube",
    "육면체": "cube",
    "정6면체": "cube",
    "큐브": "cube",
    "cube": "cube",
    "상자": "cube",
    "box": "cube",
    "네모": "cube",

    # Sphere
    "구": "sphere",
    "구체": "sphere",
    "공": "sphere",
    "sphere": "sphere",
    "ball": "sphere",
    "둥근것": "sphere",
    "동그라미": "sphere",

    # Star
    "별": "star",
    "별모양": "star",
    "스타": "star",
    "star": "star",
    "오각형별": "star",

    # Heart
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

APPROACH_CMD = {
    "tetrahedron": "APPROACH_TETRAHEDRON",
    "cube": "APPROACH_CUBE",
    "sphere": "APPROACH_SPHERE",
    "star": "APPROACH_STAR",
    "heart": "APPROACH_HEART",
}

# ==================================================
# QWEN LLM RUNNER FOR TURTLEBOT3 (Ollama)
# ==================================================
class QwenTb3:
    def __init__(
        self,
        prompt_path: str,
        model: str = "qwen2.5:14b",
        command_queue: Queue | None = None,
    ):
        self.command_queue = command_queue if command_queue else Queue()

        # prompt 로드
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_instruction = yaml.safe_load(f)["template"]

        self.model_name = model

        self.thread = None
        self.stop_event = threading.Event()

    # ------------------------------------------------
    def _extract_target_from_question(self, q: str) -> str | None:
        q_low = q.lower()
        keys = sorted(TARGET_MAP.keys(), key=len, reverse=True)
        for k in keys:
            if k.lower() in q_low:
                return TARGET_MAP[k]
        return None

    # ------------------------------------------------
    def run_qwen(self, question: str, detection_json: str) -> str:
        print(f"[QwenTb3] Using model: {self.model_name}")

        user_content = f"""
# Observation (JSON):
{detection_json}

# Question:
{question}
"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": 0.1},
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Qwen Error: {e}"

    # ------------------------------------------------
    def talk(self, sim):
        while not self.stop_event.is_set():
            try:
                question = input("\n💬 Human: ")

                # YOLO 결과
                det_dict = sim.yolo_detect_dict() or {}
                det_json = json.dumps(det_dict, ensure_ascii=False, indent=2)

                # 목표 추출
                target = self._extract_target_from_question(question)

                # ----------------------------------
                # 1️⃣ 목표 있는데 화면에 없으면 → SEARCH
                # ----------------------------------
                if (target and target not in det_dict) or (target in det_dict):
                # if target or target not in det_dict:
                    if any(k in question for k in ["찾아", "보여"]):
                        cmd = SEARCH_CMD[target]
                        print(f"➡️ '{target}' 안 보여서 {cmd} 수행")
                        self.command_queue.put(cmd)
                        continue

                # ----------------------------------
                # 2️⃣ 목표 보이고, 접근 요청이면 → APPROACH
                # ----------------------------------
                if target and target in det_dict:
                    if any(k in question for k in ["가까이", "다가가", "접근", "앞으로"]):
                        cmd = APPROACH_CMD[target]
                        print(f"➡️ 접근 요청 → {cmd}")
                        self.command_queue.put(cmd)
                        continue

                # ----------------------------------
                # 3️⃣ LLM 호출
                # ----------------------------------
                answer = self.run_qwen(question, det_json)
                print(f"\n🤖 Qwen:\n{answer}\n")

                # Action 파싱
                action_match = re.search(r"Action:\s*([^\n]+)", answer)
                action = action_match.group(1).strip() if action_match else ""

                # ----------------------------------
                # 4️⃣ 잡기 방어 로직 (Gemini와 동일)
                # ----------------------------------
                # ✅ 바로 전달
                if action == "잡기":
                    self.command_queue.put("잡기")
                    continue

                # ----------------------------------
                # 5️⃣ Action 전달
                # ----------------------------------
                if action:
                    print(f"➡️ Action 전달: {action}")
                    self.command_queue.put(action)

            except EOFError:
                break

    # ------------------------------------------------
    def start(self, sim):
        self.thread = threading.Thread(
            target=self.talk, args=(sim,), daemon=True
        )
        self.thread.start()
