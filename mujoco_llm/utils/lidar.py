import mujoco as mj
import numpy as np
from typing import List, Tuple

class Lidar:
    """
    MuJoCo 시뮬레이션에서 LiDAR 데이터를 처리하고 시각화하기 위한 유틸리티 클래스.
    모든 메서드는 정적(static)으로 제공됩니다.
    """

    @staticmethod
    def get_lidar_ranges(model: mj.MjModel, data: mj.MjData, sensor_name: str) -> np.ndarray:
        """
        replicate 태그로 생성된 360개의 센서 데이터를 모두 읽어옵니다.
        """
        first_sensor_name = f"{sensor_name}000"
        try:
            # 첫 번째 센서('laser000')의 시작 주소만 찾습니다.
            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, first_sensor_name)
            start_addr = model.sensor_adr[sensor_id]
            
            # 광선 개수를 360개로 직접 지정합니다.
            num_rays = 360
            
            # 시작 주소부터 360개의 데이터를 한 번에 읽어옵니다.
            return data.sensordata[start_addr : start_addr + num_rays].copy()
            
        except mj.FatalError:
            return np.array([])

    @staticmethod
    def calculate_lidar_world_points(
        model: mj.MjModel, data: mj.MjData, ranges: np.ndarray, sensor_name: str
    ) -> List[np.ndarray]:
        """
        거리 측정값(ranges)을 월드 좌표계의 3D 포인트로 변환합니다.
        """
        points = []
        first_sensor_name = f"{sensor_name}000"
        try:
            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, first_sensor_name)

            if hasattr(model, 'sensor_siteid'):
                first_site_id = model.sensor_siteid[sensor_id]
            else:
                if model.sensor_objtype[sensor_id] == mj.mjtObj.mjOBJ_SITE:
                    first_site_id = model.sensor_objid[sensor_id]
                else:
                    return []
            
            num_rays = len(ranges)
            for i in range(num_rays):
                current_site_id = first_site_id + i
                site_pos = data.site_xpos[current_site_id]
                site_mat = data.site_xmat[current_site_id].reshape(3, 3)
                ray_direction_local = np.array([0, 0, 1])
                ray_direction_world = site_mat @ ray_direction_local
                distance = ranges[i]
                
                if distance < model.sensor_cutoff[sensor_id]:
                    point = site_pos + distance * ray_direction_world
                    points.append(point)
            
            return points
        except mj.FatalError:
            return []