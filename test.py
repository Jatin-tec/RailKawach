import cv2
import mmaction2

# Load the YOLO model
yolo_model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load the MMAction2 model
mmaction2_model = mmaction2.models.load_model("mmaction2_model.pth")

# Start a video capture device
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        break

    # Perform object detection on the current frame using the YOLO model
    detections = yolo_model.detect(frame)

    # Extract the bounding boxes of the detected objects
    bounding_boxes = []
    for detection in detections:
        bounding_box = detection[0:4]
        bounding_boxes.append(bounding_box)

    # Crop the detected objects from the frame
    cropped_objects = []
    for bounding_box in bounding_boxes:
        cropped_object = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        cropped_objects.append(cropped_object)

    # Send the batch crops of detected objects to the MMAction2 model for action recognition
    predicted_actions = mmaction2_model.predict(cropped_objects)

    # Display the predicted actions on the frame
    for i in range(len(predicted_actions)):
        predicted_action = predicted_actions[i]
        cv2.putText(frame, predicted_action, (bounding_boxes[i][0], bounding_boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture device
cap.release()

# Destroy all windows
cv2.destroyAllWindows() 