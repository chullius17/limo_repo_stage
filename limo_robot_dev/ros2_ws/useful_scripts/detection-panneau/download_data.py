from roboflow import Roboflow

rf = Roboflow(api_key="hlkzhJgF5znE2y3tFjxI")
project = rf.workspace("prda3a").project("robot_limo-v4")
version = project.version(2)
dataset = version.download("yolov8")

print(f"\n✅ Dataset téléchargé dans : {dataset.location}")                