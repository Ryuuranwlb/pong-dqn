import numpy as np
import cv2
from collections import namedtuple, deque

class ObsProcessTool:
    def __init__(self, skip_frame=4, horizon=4, clip=True, flip=False):     # horizon：存储处理帧的数量
        self.skip_frame = skip_frame     # skip_frame：每隔多少帧处理一次图像
        self.clip = clip     # clip：是否裁剪图像帧
        self.flip = flip     # flip：是否水平翻转图像帧

        self._width = 84
        self._height = 84     # 设置处理图像的宽度和高度为 84 像素

        self.buffer = np.zeros((horizon, self._height, self._width), dtype=np.float32)
        self.obs_buffer = deque(maxlen=2)
        self.frame_cnt = -1

        self.pre_obs = np.zeros((horizon, self._height, self._width), dtype=np.float32)
        self.obs = np.zeros((horizon, self._height, self._width), dtype=np.float32)

    def process(self, obs):     # 用于处理单帧图像 obs
        obs = obs.copy()
        obs_mask = cv2.inRange(obs, np.array([213, 130, 74]), np.array([213, 130, 74]))     # 创建一个颜色掩码 obs_mask，选中像素值在 [213, 130, 74] 范围内的像素
        obs[obs_mask > 0] = [92, 186, 92]     # 将 obs_mask 选中的像素更改为颜色 [92, 186, 92]
        self.obs_buffer.append(obs)
        self.frame_cnt += 1
    
        if self.frame_cnt % self.skip_frame == 0:     # 每隔 skip_frame 帧进行一次图像处理。将 frame_cnt 重置为 0，并计算 obs_buffer 内所有图像的逐像素最大值，生成 max_frame
            self.frame_cnt = 0
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)

            if self.clip:     # 如果 clip 为 True，裁剪 max_frame，保留从垂直方向第 34 行到第 194 行（共 160 行）以及水平方向的前 160 列
                max_frame = max_frame[34:34 + 160, :160]
            if self.flip:     # 如果 flip 为 True，水平翻转 max_frame，可训练左边的板子
                max_frame = cv2.flip(max_frame, 1)

            frame = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)     # 将 max_frame 转换为灰度图像
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)     # 将灰度图像 frame 缩放到指定的宽度和高度（84x84）
            frame = frame[:, :, None]     # 添加一个新维度，使 frame 的形状从 (height, width) 变为 (height, width, 1)

            frame = np.moveaxis(frame, 2, 0)

            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = frame

            self.pre_obs = self.obs
            self.obs = np.array(self.buffer).astype(np.float32) / 255.0
            
            return 0, self.obs
        else:
            return -1, None

    def reset(self):     # 重置所有参数，包括清空 obs_buffer、重置 frame_cnt 和将 buffer、pre_obs、obs 填充为零
        self.frame_cnt = -1
        self.obs_buffer.clear()
        self.buffer.fill(0)
        self.pre_obs.fill(0)
        self.obs.fill(0)
