
from typing import List
from ultralytics.engine.results import Results

import yaml
from ultralytics import YOLO


with open("config.yaml", "r") as ff:
    config = yaml.load(ff, Loader=yaml.SafeLoader)

model = YOLO(config["model"]["best"], task="detect")


def output_image(image, confidence) -> List[Results]:
    output: List[Results] = model.predict(image, imgsz=(720,1280), conf=confidence, max_det=1, half=True) #, vid_stride=1, stream=True)
    return output

def parse_label(contents: Results):
    """
    Assume `len(list(content.boxes)) > 0` - You should pass model.predict(image)[0] to this method.
    """
    box = contents.boxes[0]
    detected_class = contents.names[box.cls.item()]
    box_coordinates = box.xyxy.int().tolist()[0]
    return {"class_name": detected_class,
            "box": {
                "x1": box_coordinates[0],
                "y1": box_coordinates[1],
                "x2": box_coordinates[2],
                "y2": box_coordinates[3]
            },
            "class_index": box.cls.int().item()}
