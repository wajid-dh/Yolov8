import cv2
import random
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # model size nano

# Random color without shades of red
def get_random_color():
    while True:
        color = [random.randint(0, 255) for _ in range(3)]
        if color[0] < 100 or (color[1] > 100 and color[2] > 100): 
            return tuple(color)

# webcam
cap = cv2.VideoCapture(0)

# Initialize variables for bounding boxes, colors, and tracking
boxes = []
colors = {}
selected_box_id = -1
start_time = 0

# OpenCV window
cv2.namedWindow('YOLOv8 Object Detection')

# click event function
def click_event(event, x, y, flags, param):
    global selected_box_id, start_time
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if selected_box_id != -1:
                    colors[boxes[selected_box_id]] = get_random_color()
                selected_box_id = i
                colors[boxes[selected_box_id]] = (0, 0, 255)
                start_time = time.time()

# Set mouse callback
cv2.setMouseCallback('YOLOv8 Object Detection', click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # objects detection in frame
    results = model(frame)
    boxes = []
    for result in results:
        for bbox, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            # Class ID = 0 is for person in COCO dataset
            if int(class_id) == 0:  
                x1, y1, x2, y2 = map(int, bbox[:4])
                boxes.append((x1, y1, x2, y2))
                if (x1, y1, x2, y2) not in colors:
                    colors[(x1, y1, x2, y2)] = get_random_color()
    
    # Draw bounding boxes and user interaction
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[(x1, y1, x2, y2)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if selected_box_id == i:
            cv2.putText(frame, f'Time: {int(time.time() - start_time)}s', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('YOLOv8 Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
