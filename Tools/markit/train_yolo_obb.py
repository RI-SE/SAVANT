from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/fredrik/fwrise/SAVANT/Prototypes/markit/datasets/UAV_yolo_obb/UAV.yaml", epochs=50, imgsz=640,
                      batch=30, device="cuda")

