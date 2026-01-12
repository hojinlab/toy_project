import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from utils.lidar import Lidar

class LidarVisualizer:
    def __init__(self, max_range=3.0):
        self.max_range = max_range
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], s=5, c='r')
        self.robot_marker = self.ax.plot(0, 0, 'go', markersize=10)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(-self.max_range, self.max_range)
        self.ax.set_title('360-degree LiDAR View (Top-Down)')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.grid(True)
        self.print_counter = 0

    def update(self, model: mj.MjModel, data: mj.MjData, sensor_name: str):
        ranges = Lidar.get_lidar_ranges(model, data, sensor_name)
        if ranges.size == 0:
            return

        # --- [진단 코드] 센서 값 확인 ---
        # if self.print_counter % 100 == 0: # 100 프레임마다 한 번씩 출력
        #     print("-" * 30)
        #     print(f"총 감지된 광선 수: {len(ranges)}")
        #     print(f"측정된 거리 (최소): {np.min(ranges):.4f} m")
        #     print(f"측정된 거리 (최대): {np.max(ranges):.4f} m")

        try:
            first_sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, f"{sensor_name}000")
            cutoff = model.sensor_cutoff[first_sensor_id]
        except mj.FatalError:
            cutoff = self.max_range
        
        # if self.print_counter % 100 == 0:
        #     print(f"센서의 최대 감지 거리 (Cutoff): {cutoff:.4f} m")

        # 최대 감지 거리(cutoff)보다 작은, 유효한 측정값만 필터링
        valid_indices = ranges < cutoff
        num_valid_points = np.sum(valid_indices)

        # if self.print_counter % 100 == 0:
        #     print(f"유효한 점의 개수: {num_valid_points}")
        #     print("-" * 30)
        
        self.print_counter += 1
        # --- [진단 코드 끝] ---

        if num_valid_points == 0:
            # 유효한 점이 하나도 없으면, 화면의 모든 점을 지웁니다.
            self.scatter.set_offsets(np.empty((0, 2)))
        else:
            num_rays = len(ranges)
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
            
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]

            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)
            
            self.scatter.set_offsets(np.c_[x_coords, y_coords])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.ioff()
        plt.close(self.fig)