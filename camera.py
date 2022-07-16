import cv2
import time
import numpy as np
import pyrealsense2 as rs

class Camera:
    def __init__(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(
                rs.stream.color, 1280, 720, rs.format.rgb8, 30)
            self.pipeline.start(self.config)

        except Exception as e:
            print(e)
            pass

    def take_picture(self, filename):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("No frame has been received. Something is wrong.")

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(filename, color_image)

    def __del__(self):
        self.pipeline.stop()
