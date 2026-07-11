import os
import sys


def get_project_root():
    """이 스크립트(scripts/) 기준으로 프로젝트 루트를 반환한다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


# 프로젝트 루트
PROJECT_ROOT = get_project_root()
sys.path.append(PROJECT_ROOT)

xml_path = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_main.xml")
print("Using XML:", xml_path)

from tb3_sim import TurtlebotFactorySim

if __name__ == "__main__":

    sim = TurtlebotFactorySim(xml_path=xml_path, use_yolo=False)
    sim.start()
