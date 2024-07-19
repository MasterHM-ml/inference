import cv2
import yaml
import base64
import requests
from utils import annotate_image, InferenceResponse, BoundBox

with open("config.yaml", "r") as ff:
    config = yaml.load(ff, yaml.SafeLoader)
app = config["app"]

vid = cv2.VideoCapture("./sample.mp4")
vid.set(cv2.CAP_PROP_POS_FRAMES, 300)

status, frame = vid.read()

while status:
    frame = cv2.resize(frame, (640,360), interpolation=cv2.INTER_LINEAR)
    response = requests.post(f"{app['url']}:{app['port']}/{app['endpoint']}",
                             files={"image": base64.b64encode(cv2.imencode(".jpg", frame)[1].tobytes()).decode()})
    if response.status_code!=200:
        raise ConnectionError(response.content.decode())
    response = response.json()
    if response["class_name"]:
        image = annotate_image(InferenceResponse(class_name=response["class_name"],
                                                 class_index=response["class_index"],
                                                 box=BoundBox(response["box"]), 
                                                 image=frame))
        cv2.imshow("img", image)
    else:
        cv2.imshow("img", frame)

    if cv2.waitKey(int(1000//vid.get(cv2.CAP_PROP_FPS))) and 0xff == "q":
        print("breaking")
        break
    status, frame = vid.read()
vid.release()
cv2.destroyAllWindows()