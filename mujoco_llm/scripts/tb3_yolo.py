import os
import sys


def get_project_root():
    """이 스크립트(scripts/) 기준으로 프로젝트 루트를 반환한다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


PROJECT_ROOT = get_project_root()

# 프로젝트 루트 추가
sys.path.append(PROJECT_ROOT)

# 통합 시뮬 클래스 import
from tb3_sim import TurtlebotFactorySim

# XML, YOLO weight 경로
XML_PATH = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_main.xml")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "scripts", "best_mac.pt")  # 위치에 맞게 수정

print("XML:", XML_PATH)
print("YOLO:", YOLO_WEIGHTS)

# 터틀봇 + YOLO 통합 시뮬 실행

# tb3_sim:
# - MuJoCo 로드
# - MuJoCoViewer 생성
# - latest_frame 업데이트
# - YOLO 로드 & OpenCV 창 띄우기까지 다 처리
if __name__ == "__main__":

    sim = TurtlebotFactorySim(
        xml_path=XML_PATH,
        use_yolo=True,              # YOLO 함께 사용
        yolo_weight_path=YOLO_WEIGHTS,
        yolo_conf=0.5,
    )

    sim.start()   # 내부에서 while 루프 + 렌더링 + YOLO + cv2.imshow("Robot YOLO View") 수행
