import torch
import cv2

def detect_and_mark_human_yolo(image_path):
    """
    Detects humans in an image using a pre-trained YOLOv5 model,
    marks the detected areas, and saves the output image.

    Args:
        image_path (str): The path to the input image.
    """
    try:
        # Load the pre-trained YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        return f"Error loading YOLOv5 model: {e}. Please check your internet connection or install PyTorch correctly."

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image could not be loaded. Please check the image path."

    # Perform detection
    results = model(image)
    
    # Extract detection information
    detections = results.pred[0]

    human_detected = False
    
    # Loop through all detected objects
    for *box, conf, cls in detections:
        # Class 0 corresponds to 'person' in the COCO dataset
        if int(cls) == 0: 
            human_detected = True
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Draw a rectangle around the person
            color = (0, 0, 255)  # Red color (BGR format)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add a label
            label = f'Person: {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
    # Save the output image if humans were detected --> 
    if human_detected:
        output_path = "yolo_output_image.jpg"
        cv2.imwrite(output_path, image)
        return f"Human(s) detected. Output image saved to '{output_path}'."
    else:
        return "No humans were detected in the image."

if __name__ == "__main__":
    # Path to your image file
    image_file = "room_image_not_human5.jpg" 
    detection_result = detect_and_mark_human_yolo(image_file)
    print(detection_result)