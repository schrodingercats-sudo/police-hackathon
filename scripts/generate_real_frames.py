"""Download real traffic camera images from public sources for demo.

Fetches actual Indian road/traffic images from public domain sources
and saves them as camera frames to replace synthetic renders.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
from pathlib import Path

# Real public traffic camera / road images from Indian cities (public domain / CC0)
REAL_TRAFFIC_SOURCES = [
    # Placeholder: We'll use OpenCV to create photorealistic composites from real road templates
    # For the hackathon demo, we'll download sample frames from public datasets
]

# Real Indian road/traffic scene URLs (creative commons / public domain sample images)
SAMPLE_URLS = {
    "CAM_DEL_001": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Connaught_Place_Delhi.jpg/1280px-Connaught_Place_Delhi.jpg",
    "CAM_DEL_002": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/ITO_Delhi.jpg/1280px-ITO_Delhi.jpg",
    "CAM_DEL_003": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Akshardham_Delhi.jpg/1280px-Akshardham_Delhi.jpg",
    "CAM_NOIDA_001": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/DND_Flyway.jpg/1280px-DND_Flyway.jpg",
    "CAM_NOIDA_002": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Noida_Expressway.jpg/1280px-Noida_Expressway.jpg",
    "CAM_GRNOIDA_001": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Pari_Chowk.jpg/1280px-Pari_Chowk.jpg",
}

def create_realistic_traffic_frame(camera_id, camera_name, output_path, width=1280, height=720):
    """Create a realistic-looking traffic surveillance frame with CCTV overlay.
    
    Uses a real-looking gradient background with road markings, vehicles,
    and professional CCTV HUD overlay. Much better than simple rectangles.
    """
    from datetime import datetime
    
    # Create a realistic road scene using OpenCV
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (realistic twilight/day)
    for y in range(height // 3):
        ratio = y / (height // 3)
        r = int(80 + ratio * 40)
        g = int(120 + ratio * 50)
        b = int(180 + ratio * 30)
        frame[y, :] = [b, g, r]
    
    # Road surface (dark asphalt with texture)
    road_top = height // 3
    for y in range(road_top, height):
        ratio = (y - road_top) / (height - road_top)
        gray = int(45 + ratio * 25 + np.random.randint(-3, 4))
        frame[y, :] = [gray, gray, gray]
    
    # Add road lane markings (white dashed lines)
    lane_y = height // 2
    for x in range(0, width, 80):
        if (x // 80) % 2 == 0:
            cv2.rectangle(frame, (x, lane_y - 2), (x + 40, lane_y + 2), (200, 200, 200), -1)
    
    # Center divider (yellow)
    cv2.rectangle(frame, (0, lane_y - 50), (width, lane_y - 46), (30, 180, 220), -1)
    
    # Add perspective road edges
    pts_left = np.array([[0, height], [width//3, road_top + 80]], dtype=np.int32)
    pts_right = np.array([[width, height], [2*width//3, road_top + 80]], dtype=np.int32)
    cv2.line(frame, tuple(pts_left[0]), tuple(pts_left[1]), (150, 150, 150), 2)
    cv2.line(frame, tuple(pts_right[0]), tuple(pts_right[1]), (150, 150, 150), 2)
    
    # Draw realistic vehicle shapes at various positions
    vehicles = [
        # (x, y, w, h, color)
        (200, height - 250, 120, 60, (180, 180, 190)),   # Silver car
        (500, height - 200, 130, 65, (30, 30, 30)),       # Black SUV
        (750, height - 280, 110, 55, (200, 200, 210)),    # White car
        (350, height - 150, 140, 70, (20, 20, 150)),      # Red truck
        (900, height - 220, 115, 58, (150, 130, 30)),     # Blue car
        (100, height - 180, 100, 50, (30, 120, 30)),      # Green car
    ]
    
    for vx, vy, vw, vh, vcolor in vehicles:
        # Car body
        cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), vcolor, -1)
        # Car roof (darker)
        cv2.rectangle(frame, (vx + 15, vy - 20), (vx + vw - 15, vy), 
                      tuple(max(0, c - 40) for c in vcolor), -1)
        # Windshield
        cv2.rectangle(frame, (vx + 20, vy - 18), (vx + vw - 20, vy - 5),
                      (80, 60, 50), -1)
        # Headlights
        cv2.circle(frame, (vx + 8, vy + vh - 10), 5, (200, 220, 255), -1)
        cv2.circle(frame, (vx + vw - 8, vy + vh - 10), 5, (200, 220, 255), -1)
        # Shadow
        cv2.rectangle(frame, (vx + 5, vy + vh), (vx + vw - 5, vy + vh + 8), (20, 20, 20), -1)
        
        # License plate (realistic yellow Indian plate)
        plate_x = vx + vw // 2 - 25
        plate_y = vy + vh - 18
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 50, plate_y + 12), (0, 200, 255), -1)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 50, plate_y + 12), (0, 0, 0), 1)
        
        # Plate text - use REAL-looking format
        plates = ["DL 1C 1234", "HR 26 5678", "UP 16 9012", "MH 12 3456", "KA 51 7890", "GJ 01 2345"]
        plate_text = plates[vehicles.index((vx, vy, vw, vh, vcolor)) % len(plates)]
        cv2.putText(frame, plate_text, (plate_x + 2, plate_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Add trees/poles on sides
    for tx in [50, 1150, 1220]:
        cv2.rectangle(frame, (tx, road_top - 100), (tx + 8, road_top + 150), (40, 80, 40), -1)
        cv2.circle(frame, (tx + 4, road_top - 100), 35, (30, 100, 30), -1)
    
    # Traffic light pole
    cv2.rectangle(frame, (width - 100, road_top - 60), (width - 95, road_top + 50), (80, 80, 80), -1)
    cv2.circle(frame, (width - 97, road_top - 40), 6, (0, 0, 200), -1)  # Red light
    cv2.circle(frame, (width - 97, road_top - 25), 6, (0, 180, 0), -1)  # Green light
    
    # CCTV HUD Overlay (professional look)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Top-left: Camera ID + timestamp
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 65), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, f"[{camera_id}]", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, camera_name[:45], (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{timestamp} IST", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 200, 100), 1, cv2.LINE_AA)
    
    # Top-right: REC indicator
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (width - 130, 0), (width, 30), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)
    cv2.circle(frame, (width - 115, 15), 6, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (width - 103, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "25fps", (width - 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Bottom crosshair grid lines (surveillance style)
    cv2.line(frame, (width // 2, height - 30), (width // 2, height), (0, 255, 0), 1)
    cv2.line(frame, (width // 2 - 15, height - 15), (width // 2 + 15, height - 15), (0, 255, 0), 1)
    
    # Border frame
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    
    # Add some noise for realism
    noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"  [+] Generated realistic frame: {output_path}")
    return True


def download_or_generate_frames(evidence_dir):
    """Generate realistic traffic camera frames for all 6 cameras."""
    frames_dir = Path(evidence_dir) / "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    cameras = {
        "CAM_DEL_001": "Connaught Place - Radial Road 1 / Outer Circle",
        "CAM_DEL_002": "ITO Junction - Vikas Minar Crossing",
        "CAM_DEL_003": "Akshardham Setu - NH24 / Delhi-Meerut Expy",
        "CAM_NOIDA_001": "DND Flyway Toll Plaza - Delhi-Noida Border",
        "CAM_NOIDA_002": "Noida-Greater Noida Expressway Sector 62",
        "CAM_GRNOIDA_001": "Pari Chowk Roundabout - Greater Noida",
    }
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for cam_id, cam_name in cameras.items():
        output_path = frames_dir / f"{cam_id}_{timestamp}.jpg"
        
        # Try to download real image first
        url = SAMPLE_URLS.get(cam_id)
        downloaded = False
        
        if url:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req, timeout=5)
                data = response.read()
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Resize to 1280x720
                    img = cv2.resize(img, (1280, 720))
                    # Add CCTV HUD overlay
                    add_cctv_overlay(img, cam_id, cam_name)
                    cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    print(f"  [+] Downloaded real frame for {cam_id}")
                    downloaded = True
            except Exception as e:
                print(f"  [!] Download failed for {cam_id}: {e}")
        
        if not downloaded:
            create_realistic_traffic_frame(cam_id, cam_name, output_path)


def add_cctv_overlay(frame, camera_id, camera_name):
    """Add professional CCTV HUD overlay to a real image."""
    from datetime import datetime
    h, w = frame.shape[:2]
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Top-left overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"[{camera_id}]", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, camera_name[:45], (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{timestamp} IST", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 200, 100), 1, cv2.LINE_AA)
    
    # Top-right REC
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (w - 130, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.circle(frame, (w - 115, 15), 6, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (w - 103, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)


if __name__ == "__main__":
    evidence_dir = Path(__file__).parent.parent / "data" / "evidence"
    print("Generating realistic traffic camera frames...")
    download_or_generate_frames(str(evidence_dir))
    print("Done!")
