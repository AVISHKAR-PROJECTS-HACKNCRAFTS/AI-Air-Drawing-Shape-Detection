import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hand_landmarks(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def get_finger_tip(self, results, width, height):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks
                lm = hand_landmarks.landmark
                
                # Finger states (Up or Down)
                # Tips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
                # Lower joints: 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
                
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                ring_up = lm[16].y < lm[14].y
                pinky_up = lm[20].y < lm[18].y
                
                # Drawing gesture: Index up, others down
                if index_up and not middle_up and not ring_up and not pinky_up:
                    cx, cy = int(lm[8].x * width), int(lm[8].y * height)
                    return cx, cy, hand_landmarks
        return None, None, None

def preprocess_gesture(points, canvas_size=(128, 128)):
    """
    Preprocesses the drawn points into a format suitable for the CNN+ViT model.
    """
    if not points:
        return None
    
    # Create a blank black image
    mask = np.zeros((480, 640), dtype=np.uint8)
    
    # Draw the points as lines (Gesture Contour Detection)
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            cv2.line(mask, points[i-1], points[i], 255, 5)
    
    # Find contours to isolate the shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get the largest contour (the shape)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Crop and resize
    roi = mask[y:y+h, x:x+w]
    roi = cv2.resize(roi, canvas_size)
    
    # Normalize
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1) # Add channel dimension
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    
    return roi

def heuristic_classify(points):
    """
    Heuristic classification for demo purposes (untrained model).
    Uses contour approximation and geometric properties.
    """
    if not points or len(points) < 10:
        return "Circle", 0.5
    
    mask = np.zeros((480, 640), dtype=np.uint8)
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            cv2.line(mask, points[i-1], points[i], 255, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Circle", 0.5
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    
    if area < 100:
        return "Circle", 0.5
        
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    num_vertices = len(approx)
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    
    # Convex Hull for Star detection
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 1
    
    # 1. Star Detection (Low solidity)
    if solidity < 0.6:
        return "Star", 0.95
        
    # 2. Triangle
    if num_vertices == 3:
        return "Triangle", 0.9
        
    # 3. Square and Rectangle
    if num_vertices == 4:
        if 0.85 <= aspect_ratio <= 1.15:
            return "Square", 0.9
        else:
            return "Rectangle", 0.9
            
    # 4. Pentagon
    if num_vertices == 5:
        return "Pentagon", 0.85
        
    # 5. Hexagon
    if num_vertices == 6:
        return "Hexagon", 0.85
        
    # 6. Heptagon
    if num_vertices == 7:
        return "Heptagon", 0.85
        
    # 7. Octagon
    if num_vertices == 8:
        return "Octagon", 0.85
        
    # 8. Circle vs Ellipse (High vertex count or smooth)
    if num_vertices > 8:
        if 0.8 <= aspect_ratio <= 1.2:
            return "Circle", 0.9
        else:
            return "Ellipse", 0.9
            
    # Default fallback based on vertices
    if num_vertices > 8: return "Circle", 0.6
    return "Circle", 0.5

def draw_perfect_shape(image, shape_name, color, points):
    """
    Renders a clean geometric version of the detected shape.
    """
    if not points:
        return image
    
    # Calculate bounding box of original points to place the perfect shape
    pts = np.array([p for p in points if p is not None])
    if len(pts) == 0: return image
    
    x, y, w, h = cv2.boundingRect(pts)
    center = (x + w // 2, y + h // 2)
    
    overlay = image.copy()
    color_bgr = (color[2], color[1], color[0]) # RGB to BGR
    
    if shape_name == "Circle":
        radius = max(w, h) // 2
        cv2.circle(overlay, center, radius, color_bgr, -1)
    elif shape_name == "Square" or shape_name == "Rectangle":
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color_bgr, -1)
    elif shape_name == "Triangle":
        pt1 = (center[0], y)
        pt2 = (x, y + h)
        pt3 = (x + w, y + h)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(overlay, [triangle_cnt], 0, color_bgr, -1)
    elif shape_name == "Pentagon":
        # Approximate pentagon
        pts_poly = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            px = int(center[0] + (w/2) * np.cos(angle))
            py = int(center[1] + (h/2) * np.sin(angle))
            pts_poly.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_poly)], 0, color_bgr, -1)
    elif shape_name == "Hexagon":
        pts_poly = []
        for i in range(6):
            angle = i * 2 * np.pi / 6
            px = int(center[0] + (w/2) * np.cos(angle))
            py = int(center[1] + (h/2) * np.sin(angle))
            pts_poly.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_poly)], 0, color_bgr, -1)
    elif shape_name == "Heptagon":
        pts_poly = []
        for i in range(7):
            angle = i * 2 * np.pi / 7 - np.pi / 2
            px = int(center[0] + (w/2) * np.cos(angle))
            py = int(center[1] + (h/2) * np.sin(angle))
            pts_poly.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_poly)], 0, color_bgr, -1)
    elif shape_name == "Octagon":
        pts_poly = []
        for i in range(8):
            angle = i * 2 * np.pi / 8 - np.pi / 2
            px = int(center[0] + (w/2) * np.cos(angle))
            py = int(center[1] + (h/2) * np.sin(angle))
            pts_poly.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_poly)], 0, color_bgr, -1)
    elif shape_name == "Ellipse":
        cv2.ellipse(overlay, center, (w//2, h//2), 0, 0, 360, color_bgr, -1)
    elif shape_name == "Star":
        # Simple 5-pointed star
        pts_star = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            r = (w/2) if i % 2 == 0 else (w/4)
            px = int(center[0] + r * np.cos(angle))
            py = int(center[1] + r * np.sin(angle))
            pts_star.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_star)], 0, color_bgr, -1)
    else:
        # Default for others like Heptagon, Octagon (generalized polygon)
        sides = {"Heptagon": 7, "Octagon": 8}.get(shape_name, 4)
        pts_poly = []
        for i in range(sides):
            angle = i * 2 * np.pi / sides
            px = int(center[0] + (w/2) * np.cos(angle))
            py = int(center[1] + (h/2) * np.sin(angle))
            pts_poly.append([px, py])
        cv2.drawContours(overlay, [np.array(pts_poly)], 0, color_bgr, -1)

    # Transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add text
    cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image
