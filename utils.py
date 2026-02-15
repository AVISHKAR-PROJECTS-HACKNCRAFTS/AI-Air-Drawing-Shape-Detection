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
    
    # Add padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(mask.shape[1] - x, w + 2 * pad)
    h = min(mask.shape[0] - y, h + 2 * pad)
    
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
    Heuristic classification fallback for the 12 creative shapes.
    Uses geometric properties and contour analysis to make best guesses.
    """
    if not points or len(points) < 10:
        return "Spiral", 0.4
    
    mask = np.zeros((480, 640), dtype=np.uint8)
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            cv2.line(mask, points[i-1], points[i], 255, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Spiral", 0.4
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    
    if area < 100:
        return "Spiral", 0.4
    
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    num_vertices = len(approx)
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 1
    
    # Convex Hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 1
    
    # Circularity
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    
    # Count contour components (for multi-part shapes)
    all_contours = contours
    num_contours = len(all_contours)
    
    # Check if the path is open (non-closed) — characteristic of spiral, lightning, music note
    pts_array = np.array(points)
    start_end_dist = np.linalg.norm(pts_array[0] - pts_array[-1])
    max_extent = max(w, h)
    is_open = start_end_dist > max_extent * 0.3
    
    # Heuristic Decision Tree for 12 creative shapes
    
    # Smiley face: round with high circularity, multiple contour components (eyes inside)
    if circularity > 0.6 and aspect_ratio > 0.7 and aspect_ratio < 1.4:
        return "Smiley face", 0.7
    
    # Lightning bolt: very elongated, open path, zigzag pattern (low solidity)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        if is_open and solidity < 0.7:
            return "Lightning bolt", 0.7
    
    # Spiral: open path, curving, moderate solidity
    if is_open and circularity < 0.3:
        return "Spiral", 0.65
    
    # Infinity: wide aspect ratio, figure-8 like, closed
    if aspect_ratio > 1.5 and not is_open and solidity > 0.5:
        return "Infinity", 0.6
    
    # Flower: multiple bumps, moderate solidity
    if num_contours > 2 or (solidity < 0.6 and circularity > 0.2):
        return "Flower", 0.6
    
    # Butterfly: wide, symmetric, low-moderate solidity
    if aspect_ratio > 1.2 and solidity < 0.65:
        return "Butterfly", 0.55
    
    # Crown: wider than tall, zigzag top
    if aspect_ratio > 1.3 and num_vertices > 5 and num_vertices < 15:
        return "Crown", 0.55
    
    # Fish: elongated ellipse with tail
    if aspect_ratio > 1.5 and solidity > 0.6:
        return "Fish", 0.55
    
    # Leaf: elongated, pointed ends
    if aspect_ratio > 1.3 or aspect_ratio < 0.7:
        if solidity > 0.7:
            return "Leaf", 0.55
    
    # Flame: taller than wide, tapered top
    if aspect_ratio < 0.8 and solidity > 0.5:
        return "Flame", 0.55
    
    # Music note: open path, small area
    if is_open and area < hull_area * 0.5:
        return "Music note", 0.5
    
    # Cloud: round-ish, bumpy
    if circularity > 0.3 and solidity > 0.6:
        return "Cloud", 0.5
    
    # Default fallback
    return "Spiral", 0.4


def draw_perfect_shape(image, shape_name, color, points):
    """
    Renders a clean version of the detected shape overlay.
    Only handles the 12 creative shapes.
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
    
    if shape_name == "Spiral":
        pts_draw = []
        for theta in np.arange(0, 4 * np.pi, 0.1):
            r = (theta / (4 * np.pi)) * (min(w, h)/2)
            px = int(center[0] + r * np.cos(theta))
            py = int(center[1] + r * np.sin(theta))
            pts_draw.append([px, py])
        cv2.polylines(overlay, [np.array(pts_draw)], False, color_bgr, 5)
    
    elif shape_name == "Infinity":
        pts_draw = []
        for t in np.arange(0, 2 * np.pi, 0.1):
            den = 1 + np.sin(t)**2
            scale = min(w, h) / 2
            px = int(center[0] + scale * np.cos(t) / den * 1.5)
            py = int(center[1] + scale * np.sin(t) * np.cos(t) / den * 1.5)
            pts_draw.append([px, py])
        cv2.polylines(overlay, [np.array(pts_draw)], True, color_bgr, 5)
    
    elif shape_name == "Cloud":
        r = int(min(w, h) * 0.4)
        circles = [
            (center[0], center[1], r),
            (center[0]-int(r*0.8), center[1]+int(r*0.3), int(r*0.7)),
            (center[0]+int(r*0.8), center[1]+int(r*0.3), int(r*0.7)),
            (center[0], center[1]-int(r*0.5), int(r*0.6))
        ]
        for (cx, cy, cr) in circles:
            cv2.circle(overlay, (cx, cy), cr, color_bgr, -1)
    
    elif shape_name == "Lightning bolt":
        pts_draw = np.array([
            [center[0]-w//4, center[1]-h//2],
            [center[0]+w//4, center[1]-h//6],
            [center[0], center[1]-h//6],
            [center[0]+w//4, center[1]+h//2],
            [center[0]-w//4, center[1]+h//6],
            [center[0], center[1]+h//6]
        ])
        cv2.drawContours(overlay, [pts_draw], 0, color_bgr, -1)
    
    elif shape_name == "Flower":
        cv2.circle(overlay, center, min(w,h)//4, color_bgr, -1)
        for i in range(5):
            angle = i * 2 * np.pi / 5
            px = int(center[0] + (min(w,h)//2) * np.cos(angle))
            py = int(center[1] + (min(w,h)//2) * np.sin(angle))
            cv2.circle(overlay, (px, py), min(w,h)//4, color_bgr, -1)
    
    elif shape_name == "Butterfly":
        cv2.ellipse(overlay, center, (w//8, h//2), 0, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]-w//4, center[1]-h//4), (w//3, h//3), 30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]+w//4, center[1]-h//4), (w//3, h//3), -30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]-w//4, center[1]+h//4), (w//4, h//4), -30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]+w//4, center[1]+h//4), (w//4, h//4), 30, 0, 360, color_bgr, -1)
    
    elif shape_name == "Crown":
        pts_draw = np.array([
            [center[0]-w//2, center[1]+h//2],
            [center[0]-w//2, center[1]-h//4],
            [center[0]-w//4, center[1]],
            [center[0], center[1]-h//2],
            [center[0]+w//4, center[1]],
            [center[0]+w//2, center[1]-h//4],
            [center[0]+w//2, center[1]+h//2]
        ])
        cv2.drawContours(overlay, [pts_draw], 0, color_bgr, -1)
    
    elif shape_name == "Flame":
        pts_draw = []
        for t in np.arange(0, 2*np.pi, 0.1):
            fx = int(center[0] + (w/2) * np.cos(t))
            fy = int(center[1] + (h/2) * np.sin(t) * (np.sin(t/2)**0.5 if t < np.pi else 1))
            pts_draw.append([fx, fy])
        cv2.drawContours(overlay, [np.array(pts_draw)], 0, color_bgr, -1)
    
    elif shape_name == "Fish":
        cv2.ellipse(overlay, center, (w//2, h//3), 0, 0, 360, color_bgr, -1)
        tail_pts = np.array([
            [center[0]-w//2, center[1]],
            [center[0]-w, center[1]-h//4],
            [center[0]-w, center[1]+h//4]
        ])
        cv2.drawContours(overlay, [tail_pts], 0, color_bgr, -1)
    
    elif shape_name == "Leaf":
        pts_draw = []
        for t in np.linspace(0, np.pi, 20):
            pts_draw.append([int(center[0]-w//2 + w*t/np.pi), int(center[1] - (h//2)*np.sin(t))])
        for t in np.linspace(0, np.pi, 20):
            pts_draw.append([int(center[0]+w//2 - w*t/np.pi), int(center[1] + (h//2)*np.sin(t))])
        cv2.drawContours(overlay, [np.array(pts_draw)], 0, color_bgr, -1)
    
    elif shape_name == "Music note":
        cv2.circle(overlay, (center[0]-w//4, center[1]+h//4), w//6, color_bgr, -1)
        cv2.line(overlay, (center[0]-w//4 + w//6, center[1]+h//4), (center[0]-w//4 + w//6, center[1]-h//2), color_bgr, 5)
        cv2.line(overlay, (center[0]-w//4 + w//6, center[1]-h//2), (center[0]+w//4, center[1]-h//4), color_bgr, 5)
    
    elif shape_name == "Smiley face":
        cv2.circle(overlay, center, min(w,h)//2, color_bgr, 5)
        cv2.circle(overlay, (center[0]-w//6, center[1]-w//6), w//20, color_bgr, -1)
        cv2.circle(overlay, (center[0]+w//6, center[1]-w//6), w//20, color_bgr, -1)
        cv2.ellipse(overlay, (center[0], center[1]+w//10), (w//4, w//6), 0, 0, 180, color_bgr, 5)
    
    else:
        # Fallback: just draw a highlight around the drawn points
        cv2.polylines(overlay, [pts], False, color_bgr, 4)

    # Transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add text
    cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image
