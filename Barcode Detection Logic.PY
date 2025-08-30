import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import os

def process_image(image_bytes: bytes, model_path: str):
    """
    Processes an image to detect and decode barcodes.

    Args:
        image_bytes: The image data as bytes.
        model_path: The path to the YOLO model file.

    Returns:
        A set of unique, decoded barcode data strings.
    """
    # Convert image bytes to a numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image.")

    # Load the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading model: {e}")

    # Perform detection
    results = model(img, verbose=False)
    
    seen_barcodes = set()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        
        # Extract the region of interest (ROI) for the detected barcode
        barcode_roi = img[y1:y2, x1:x2]
        
        # Decode the barcode from the ROI
        decoded_objects = decode(barcode_roi)
        
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                seen_barcodes.add(data)
                
    return seen_barcodes
