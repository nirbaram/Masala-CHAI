import numpy as np
import easyocr
import cv2

# Function to inpaint the text
def inpaint_text(image, pipeline):
    # Recognize text (and corresponding regions)
    prediction_groups = pipeline.recognize([image])

    # Define the mask for inpainting
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        # Extract the coordinates of the bounding box around the text
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        # Create a polygon that covers the text area more precisely
        polygon = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Fill the polygon area on the mask
        cv2.fillPoly(mask, [polygon], 255)

    # Inpaint the image to remove text
    inpainted_img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)

    return inpainted_img

# Class to handle text removal from images
class TextRemover:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def remove_text(self, img_path):
        # Read the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect text
        results = self.reader.readtext(img)
        
        # Create mask for inpainting
        mask = np.zeros(img.shape[:2], dtype="uint8")
        
        # Draw detected text regions on mask
        for (bbox, text, prob) in results:
            points = np.array([[int(x), int(y)] for x, y in bbox])
            cv2.fillPoly(mask, [points], 255)
        
        # Inpaint
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        
        return inpainted_img

