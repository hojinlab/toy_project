import os
import cv2
from tqdm import tqdm

class VideoToFramesConverter:
    def __init__(self, video_path, output_dir, saving_fps=4):
        self.video_path = video_path
        self.output_dir = output_dir
        self.saving_fps = saving_fps
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"프레임 저장 경로 : {self.output_dir}")

    def save_frames(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise FileNotFoundError(f"Video is unavailable : {self.video_path}")

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"비디오 프레임의 총 프레임 수 : {length}")
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"초당 프레임 수 : {fps}")
        print(f"저장할 프레임의 fps : {self.saving_fps}")

        count = 0
        frame_idx = 0
        saving_interval = max(int(fps / self.saving_fps), 1)

        with tqdm(total=length, desc="Processing frames") as pbar:
            while video.isOpened():
                ret, image = video.read()
                if not ret:
                    break

                if frame_idx % saving_interval == 0:
                    frame_path = os.path.join(self.output_dir, f"frame{count}.jpg")
                    cv2.imwrite(frame_path, image)
                    count += 1

                frame_idx += 1
                pbar.update(1)

                if frame_idx >= length:
                    break

        video.release()
        print(f"총 저장된 프레임 수 : {count}")