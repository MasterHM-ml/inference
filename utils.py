import cv2
from numpy._typing import NDArray
from dataclasses import dataclass


@dataclass
class BoundBox:
    def __init__(self, box) -> None:
        self.x1: int = box["x1"]
        self.y1: int = box["y1"]
        self.x2: int = box["x2"]
        self.y2: int = box["y2"]

@dataclass
class InferenceResponse:
    class_index: int
    image: NDArray
    class_name: str
    box: BoundBox


def annotate_image(ir: InferenceResponse):
    image = cv2.rectangle(ir.image, (ir.box.x1, ir.box.y1), (ir.box.x2, ir.box.y2),
                          thickness=5, color=(0,0,255))
    image = cv2.putText(image, ir.class_name, (ir.box.x1, ir.box.y1-10),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), thickness=2)
    return image