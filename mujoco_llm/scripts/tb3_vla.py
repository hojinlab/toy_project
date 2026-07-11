import os
from queue import Queue

from tb3_sim import TurtlebotFactorySim
from gemini_tb3 import GeminiTb3
from qwen_tb3 import QwenTb3


def get_project_root():
    """이 스크립트(scripts/) 기준으로 프로젝트 루트를 반환한다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


PROJECT_ROOT = get_project_root()
xml_path = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_main.xml")
prompt_path = os.path.join(PROJECT_ROOT, "scripts", "prompt.yaml")
yolo_weights = os.path.join(PROJECT_ROOT, "scripts", "best_mac.pt")

cmd_q = Queue()

# 1) 터틀봇 + YOLO 시뮬
sim = TurtlebotFactorySim(
    xml_path=xml_path,
    use_yolo=True,
    yolo_weight_path=yolo_weights,
    yolo_conf=0.4,
    command_queue=cmd_q,
    fps=60,
)

# 2) Gemini + YOLO + 명령 생성
# agent = GeminiTb3(
#     prompt_path=prompt_path,
#     model="gemini-robotics-er-1.5-preview",
#     command_queue=cmd_q,
# )

agent = QwenTb3(
    prompt_path=prompt_path,
    model="qwen2.5:14b",
    command_queue=cmd_q,
)

# 3) LLM 쓰레드 시작
agent.start(sim)

# 4) 시뮬 루프 시작 (키보드로 q 누르면 종료)
sim.start()