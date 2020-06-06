import cv2
import numpy as np


class Screen:

    def __init__(self, canvas: np.ndarray = None, title: str = 'Screen') -> None:
        self._title = title
        if canvas is None:
            canvas = np.zeros((480, 640), dtype=np.uint8)
        cv2.imshow(self._title, canvas)

    def update(self, canvas: np.ndarray) -> bool:
        if cv2.getWindowProperty(self._title, 0) < 0:
            return cv2.destroyWindow(self._title)
        cv2.imshow(self._title, canvas)
        cv2.waitKey(1)
        return True
