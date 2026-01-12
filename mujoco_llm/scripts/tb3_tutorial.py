import os
import sys

# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

xml_path = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_main.xml")
print("Using XML:", xml_path)

from tb3_sim import TurtlebotFactorySim

if __name__ == "__main__":

    sim = TurtlebotFactorySim(xml_path=xml_path, use_yolo=False)
    sim.start()
################