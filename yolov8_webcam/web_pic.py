from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict(source="0")  # Get all predictions

# Filter results to only include person class (assuming class label is available)
person_results = [result for result in results if result.label == "person"]

# Now you have detections only for "person" class (further processing on person_results)
