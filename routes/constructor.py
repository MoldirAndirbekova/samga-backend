# games/constructor.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
import cv2
import random
import base64
import os
import time
import math

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
PREVIEW_DURATION = 5  # seconds to show preview before game starts
SNAP_THRESHOLD = 30  # Distance threshold for snapping pieces
PLACEMENT_SCORE = 10  # Points for correctly placing a piece
TIME_BONUS_MULTIPLIER = 0.5  # Bonus points for remaining time

# Level configurations
LEVELS = [
    {"level": 1, "folder": "mushroom", "duration": 120, "speed": 0, "name": "Mushroom"},
    {"level": 2, "folder": "flower", "duration": 120, "speed": 0, "name": "Flower"},
    {"level": 3, "folder": "car", "duration": 140, "speed": 0, "name": "Car"},
    {"level": 4, "folder": "tree", "duration": 140, "speed": 0, "name": "Tree"},
    {"level": 5, "folder": "train", "duration": 140, "speed": 0, "name": "Train"},
    {"level": 6, "folder": "rainbow", "duration": 200, "speed": 0, "name": "Rainbow"},
    {"level": 7, "folder": "home", "duration": 220, "speed": 0, "name": "Home"},
    {"level": 8, "folder": "worm", "duration": 220, "speed": 0, "name": "Worm"},
    {"level": 9, "folder": "castle", "duration": 240, "speed": 0, "name": "Castle"}
]


class DragElement:
    def __init__(self, image_path, position, target_position=None, element_id=None, move_speed=0):
        self.position = list(position)
        self.target_position = target_position if target_position else list(position)
        self.image_path = image_path
        self.element_id = element_id
        self.is_dragging = False
        self.is_placed_correctly = False
        self.move_speed = move_speed
        self.image = None
        self.size = (0, 0)

        # Load image
        self.load_image()

        # Initialize velocity for moving elements
        if self.move_speed > 0:
            self.vx = random.choice([-1, 1]) * self.move_speed
            self.vy = random.choice([-1, 1]) * self.move_speed
        else:
            self.vx, self.vy = 0, 0

    def load_image(self):
        """Load and prepare the image with transparency"""
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error loading image: {self.image_path}")
                return

            # Convert to BGRA if necessary
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Scale the image
            scale_factor = 0.5
            self.image = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            self.size = self.image.shape[:2]
        except Exception as e:
            print(f"Error processing image {self.image_path}: {e}")

    def update(self, hand_position, is_pinching):
        """Update element position based on hand interaction"""
        if self.image is None or self.is_placed_correctly:
            return

        x, y = self.position
        h, w = self.size

        # Check if hand is over the element
        if x < hand_position[0] < x + w and y < hand_position[1] < y + h:
            if is_pinching:
                self.is_dragging = True

        # Update position if dragging
        if self.is_dragging:
            self.position = [hand_position[0] - w // 2, hand_position[1] - h // 2]

        # Stop dragging when pinch is released
        if not is_pinching:
            if self.is_dragging:
                # Check if element is near its target position
                distance = math.sqrt(
                    (self.position[0] - self.target_position[0]) ** 2 +
                    (self.position[1] - self.target_position[1]) ** 2
                )

                if distance < SNAP_THRESHOLD:
                    # Snap to target position
                    self.position = list(self.target_position)
                    self.is_placed_correctly = True
                    return True  # Return True when correctly placed

            self.is_dragging = False

        # Update position for moving elements
        if not self.is_dragging and self.move_speed > 0:
            self.position[0] += self.vx
            self.position[1] += self.vy

            # Bounce off walls
            if self.position[0] < 0 or self.position[0] + w > GAME_WIDTH:
                self.vx = -self.vx
            if self.position[1] < 0 or self.position[1] + h > GAME_HEIGHT:
                self.vy = -self.vy

        return False  # Not correctly placed


class ConstructorGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id

        # Game state
        self.current_level = 1
        self.selected_level = None
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        self.time_remaining = 0
        self.elements = []
        self.score = 0
        self.pieces_placed = 0
        self.total_pieces = 0
        self.showing_preview = False
        self.preview_start_time = None
        self.game_started = False

        # Hand tracking
        self.left_hand = None
        self.right_hand = None

        # Camera frame storage for AR overlay
        self.current_camera_frame = None

        # Target positions for pieces (loaded from level data)
        self.target_positions = []

        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps

   
def start_game(self, level=None):
    """Start or restart the game with selected level"""
    if level is not None:
        self.selected_level = level
    elif self.selected_level is None:
        self.selected_level = self.current_level

    print(f"Starting Constructor game id: {self.game_id} with level: {self.selected_level}")
    
    # Reset scores
    self.score = 0
    self.pieces_placed = 0

    # First show the preview
    self.showing_preview = True
    self.preview_start_time = datetime.now()
    self.game_active = False
    self.game_started = True

    # Debug log for level folders
    level_data = LEVELS[self.selected_level - 1]
    folder = level_data["folder"]
    print(f"Selected level {self.selected_level}: {level_data['name']} with folder: {folder}")
    
    # Debug possible asset paths
    possible_paths = [
        os.path.join("game_assets", "constructor", folder),
        os.path.join("assets", "constructor", folder),
        os.path.join("static", "constructor", folder),
        os.path.join(folder)
    ]
    
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"Checking path: {path} - Exists: {exists}")
        if exists:
            try:
                files = os.listdir(path)
                print(f"  Files: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
            except Exception as e:
                print(f"  Error listing directory: {e}")

    # Prepare level data
    self.load_level_data()
 
    def load_level_data(self): 
        if self.selected_level is None or self.selected_level < 1 or self.selected_level > len(LEVELS):
            print(f"Invalid level: {self.selected_level}")
        return

    level_data = LEVELS[self.selected_level - 1]
    self.time_remaining = level_data["duration"]

    # Load level elements
    folder = level_data["folder"]
    
    # Try all possible asset paths in a more systematic way
    possible_paths = [
        os.path.join("game_assets", "constructor", folder),
        os.path.join("assets", "constructor", folder),
        os.path.join("static", "constructor", folder),
        os.path.join("game_assets", "constructor", str(self.selected_level)),  # Try numeric folder
        os.path.join("assets", "constructor", str(self.selected_level)),       # Try numeric folder
        os.path.join("static", "constructor", str(self.selected_level)),       # Try numeric folder
        os.path.join(folder),  # Direct folder name
        os.path.join("constructor", folder),  # Try just constructor/folder
    ]
    
    asset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            asset_path = path
            print(f"Found assets in path: {asset_path}")
            break
    
    if not asset_path:
        print(f"ERROR: Could not find assets for level {self.selected_level} ({folder})")
        # Default to level 1 as fallback instead of returning
        self.selected_level = 1
        return self.load_level_data()  # Recursively try with level 1
    
    # Print directory contents to debug
    print(f"Contents of {asset_path}:")
    try:
        for file in os.listdir(asset_path):
            print(f"  - {file}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    # Load preview image - try multiple possible filenames
    preview_image_candidates = [
        os.path.join(asset_path, f"level{self.selected_level}.png"),
        os.path.join(asset_path, f"preview.png"),
        os.path.join(asset_path, f"{folder}.png"),
        os.path.join(asset_path, f"level{self.selected_level}_preview.png")
    ]
    
    self.preview_image_path = None
    for preview_path in preview_image_candidates:
        if os.path.exists(preview_path):
            self.preview_image_path = preview_path
            print(f"Found preview image: {self.preview_image_path}")
            break
    
    if not self.preview_image_path:
        print(f"Warning: No preview image found for level {self.selected_level}")
        # Try to find any PNG file as a fallback
        for file in os.listdir(asset_path):
            if file.lower().endswith('.png') and not file.lower().startswith("piece"):
                self.preview_image_path = os.path.join(asset_path, file)
                print(f"Using fallback preview image: {self.preview_image_path}")
                break

    # Load draggable elements
    self.elements = []
    try:
        # Look for specific element patterns in filenames
        element_files = [f for f in os.listdir(asset_path) if 
                        (not f.lower().startswith("level") and 
                         not f.lower() == "preview.png" and
                         not f.lower() == f"{folder}.png" and
                         f.lower().endswith(('.png', '.jpg', '.jpeg')))]
        
        # If no elements found, look for pieces* or part* files
        if not element_files:
            element_files = [f for f in os.listdir(asset_path) if 
                           (f.lower().startswith(("piece", "part", "element")) and 
                            f.lower().endswith(('.png', '.jpg', '.jpeg')))]
        
        print(f"Found {len(element_files)} elements in {asset_path}")

        self.total_pieces = len(element_files)
        
        if self.total_pieces == 0:
            print(f"ERROR: No element pieces found for level {self.selected_level}")
            return
            
        # Rest of the function remains unchanged...
        center_x = GAME_WIDTH // 2
        center_y = GAME_HEIGHT // 2

        # Create a simple grid pattern for target positions
        self.target_positions = []
        grid_size = int(math.ceil(math.sqrt(self.total_pieces)))
        spacing = 100

        for i in range(self.total_pieces):
            row = i // grid_size
            col = i % grid_size
            target_x = center_x - (grid_size * spacing) // 2 + col * spacing
            target_y = center_y - (grid_size * spacing) // 2 + row * spacing
            self.target_positions.append([target_x, target_y])

        for i, filename in enumerate(element_files):
            file_path = os.path.join(asset_path, filename)
            # Start position (random around edges)
            start_x = random.randint(50, GAME_WIDTH - 150)
            start_y = random.randint(50, GAME_HEIGHT - 150)

            # Target position
            target_pos = self.target_positions[i] if i < len(self.target_positions) else [center_x, center_y]

            element = DragElement(file_path, [start_x, start_y], target_pos, i, level_data["speed"])
            self.elements.append(element)
            print(f"Loaded element: {filename}")

    except Exception as e:
        print(f"Error loading elements: {e}")
        import traceback
        traceback.print_exc()

def update_game_state(self):
    """Update the game state for one frame"""
    if not self.game_started:
        return

    now = datetime.now()

    # Handle preview phase
    if self.showing_preview:
        preview_elapsed = (now - self.preview_start_time).total_seconds()
        if preview_elapsed >= PREVIEW_DURATION:
            self.showing_preview = False
            self.game_active = True
            self.start_time = now
            self.last_update = now
        return

    if not self.game_active or self.game_over:
        return

    # Update delta time
    self.last_update = now

    # Update game timer
    elapsed_seconds = (now - self.start_time).total_seconds()
    
    # Check if level data is available (fix for level issues)
    if self.selected_level is None or self.selected_level < 1 or self.selected_level > len(LEVELS):
        print(f"Fixing invalid level: {self.selected_level}")
        self.selected_level = 1  # Default to level 1
    
    self.time_remaining = max(0, LEVELS[self.selected_level - 1]["duration"] - int(elapsed_seconds))

    # Check if all pieces are placed
    if self.total_pieces > 0 and self.pieces_placed >= self.total_pieces:
        self.game_over = True
        self.game_active = False

        # Add time bonus
        time_bonus = int(self.time_remaining * TIME_BONUS_MULTIPLIER)
        self.score += time_bonus
        print(f"Level completed! Time bonus: {time_bonus}, Final score: {self.score}")

        self.save_game_result()
        return

    # Check if time is up
    if self.time_remaining <= 0:
        self.game_over = True
        self.game_active = False
        print(f"Time's up! Pieces placed: {self.pieces_placed}/{self.total_pieces}, Score: {self.score}")
        self.save_game_result()
        return

    # Check hand interactions
    self.check_hand_interactions()
    def save_game_result(self):
        """Save the game result for reporting"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()

            # Calculate skill metrics
            skill_metrics = self.calculate_skill_metrics()

            # Create game result object
            result = {
                "game_id": self.game_id,
                "difficulty": self.difficulty,
                "score": self.score,
                "duration_seconds": int(duration),
                "left_score": self.score,
                "right_score": 0,
                "timestamp": datetime.now().isoformat(),
                "skills": skill_metrics,
                "child_id": self.child_id
            }

            # Add to global results list
            from .games import game_results
            game_results.append(result)
            print(f"Game result saved - Score: {self.score}, Duration: {duration}s")

            # Save to database
            import asyncio
            asyncio.create_task(self._persist_to_database(result, skill_metrics))

            # Keep only the last 100 results
            if len(game_results) > 100:
                game_results.pop(0)

    def calculate_skill_metrics(self):
        """Calculate skill metrics based on game performance"""
        completion_rate = (self.pieces_placed / self.total_pieces) * 100 if self.total_pieces > 0 else 0
        time_efficiency = (self.time_remaining / LEVELS[self.selected_level - 1]["duration"]) * 100

        metrics = {
            "hand_eye_coordination": min(100, completion_rate),
            "spatial_reasoning": min(100, (completion_rate + time_efficiency) / 2),
            "memory": min(100, completion_rate * 0.9),  # Memory based on completion
            "creativity": min(100, self.score / (self.total_pieces * PLACEMENT_SCORE) * 100)
        }
        return metrics

    async def _persist_to_database(self, result, skill_metrics):
        """Persist game result to database using Prisma"""
        try:
            from database import prisma

            # Check if the database is connected
            if not prisma.is_connected():
                await prisma.connect()

            # Use constructor game type ID
            game_type_id = "constructor"

            # Create the game report
            game_report = await prisma.gamereport.create(
                data={
                    "gameId": result["game_id"],
                    "gameTypeId": game_type_id,
                    "childId": result["child_id"],
                    "difficulty": result["difficulty"],
                    "score": result["score"],
                    "leftScore": result["left_score"],
                    "rightScore": result["right_score"],
                    "durationSeconds": result["duration_seconds"],
                    "skillMetrics": {
                        "create": [
                            {"skillName": skill, "value": value}
                            for skill, value in skill_metrics.items()
                        ]
                    }
                },
                include={"skillMetrics": True}
            )

            print(f"Saved constructor game report to database with ID: {game_report.id}")

        except Exception as e:
            print(f"Error saving constructor game report to database: {str(e)}")

    def render_frame(self):
        """Render the current game state with AR background"""
        print(
            f"Rendering Constructor frame: active={self.game_active}, level={self.selected_level}, score={self.score}")

        # Start with camera frame as background
        if self.current_camera_frame is not None:
            try:
                # Handle base64 or raw frame
                if isinstance(self.current_camera_frame, str):
                    image_data = base64.b64decode(self.current_camera_frame.split(',')[
                                                      1] if ',' in self.current_camera_frame else self.current_camera_frame)
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    img = self.current_camera_frame

                if img is not None:
                    img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))
                else:
                    img = self._create_default_background()
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                img = self._create_default_background()
        else:
            img = self._create_default_background()

        # Apply slight overlay for better visibility
        overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        overlay.fill(30)  # Dark overlay
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        # Show preview if in preview phase
        if self.showing_preview and os.path.exists(self.preview_image_path):
            try:
                preview = cv2.imread(self.preview_image_path)
                if preview is not None:
                    # Calculate smaller preview size (50% of game screen)
                    preview_scale = 0.5
                    preview_width = int(GAME_WIDTH * preview_scale)
                    preview_height = int(GAME_HEIGHT * preview_scale)

                    # Resize preview with aspect ratio preservation
                    h, w = preview.shape[:2]
                    aspect = w / h
                    if aspect > preview_width / preview_height:
                        new_width = preview_width
                        new_height = int(preview_width / aspect)
                    else:
                        new_height = preview_height
                        new_width = int(preview_height * aspect)

                    preview_resized = cv2.resize(preview, (new_width, new_height))

                    # Calculate position to center the preview
                    x_offset = (GAME_WIDTH - new_width) // 2
                    y_offset = (GAME_HEIGHT - new_height) // 2

                    # Create a semi-transparent background for the preview
                    preview_bg = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
                    preview_bg[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = preview_resized

                    # Blend the preview with the background
                    cv2.addWeighted(preview_bg, 0.7, img, 0.3, 0, img)

                    # Add text
                    cv2.putText(img, "MEMORIZE THE PATTERN", (GAME_WIDTH // 2 - 200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                    # Show countdown
                    time_left = PREVIEW_DURATION - (datetime.now() - self.preview_start_time).total_seconds()
                    cv2.putText(img, f"Starting in: {int(time_left)}s", (GAME_WIDTH // 2 - 100, GAME_HEIGHT - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error loading preview: {e}")

        # Draw game elements
        elif self.game_active:
            # Draw target positions
            for i, target_pos in enumerate(self.target_positions):
                if i < len(self.elements) and not self.elements[i].is_placed_correctly:
                    # Draw target outline
                    cv2.circle(img, (int(target_pos[0]), int(target_pos[1])), 30, (255, 255, 255), 2)

            # Draw draggable elements
            for element in self.elements:
                if element.image is not None:
                    self._overlay_png(img, element.image, element.position)

                    # Draw highlight for correctly placed pieces
                    if element.is_placed_correctly:
                        x, y = element.position
                        h, w = element.size
                        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)

            # Draw hand indicators
            self._draw_hand_indicators(img)

            # Draw HUD
            self._draw_hud(img)

        # Show level selection if game not started
        elif not self.game_started:
            self._draw_level_selection(img)

        # Game over screen
        if self.game_over:
            self._draw_game_over(img)

        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"

        return None

    def _create_default_background(self):
        """Create a default background"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        # Create gradient
        for y in range(GAME_HEIGHT):
            color_value = int(100 + (y / GAME_HEIGHT) * 100)
            cv2.line(img, (0, y), (GAME_WIDTH, y), (color_value, color_value, 150), 1)
        return img

    def _overlay_png(self, background, overlay, position):
        """Overlay a PNG image with transparency onto background"""
        x, y = position
        h, w = overlay.shape[:2]

        # Ensure overlay fits within background
        if x + w > background.shape[1]:
            w = background.shape[1] - x
        if y + h > background.shape[0]:
            h = background.shape[0] - y
        if x < 0:
            overlay = overlay[:, -x:]
            w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h = overlay.shape[0]
            y = 0

        if w <= 0 or h <= 0:
            return

        # Extract alpha channel
        if overlay.shape[2] == 4:
            alpha = overlay[:h, :w, 3] / 255.0
            for c in range(3):
                background[y:y + h, x:x + w, c] = (
                        alpha * overlay[:h, :w, c] +
                        (1 - alpha) * background[y:y + h, x:x + w, c]
                )

    def _draw_hand_indicators(self, img):
        """Draw hand position indicators"""
        scale_x = GAME_WIDTH / 640
        scale_y = GAME_HEIGHT / 480

        for hand in [self.left_hand, self.right_hand]:
            if hand and "index_finger_tip" in hand:
                hand_x = int(hand["index_finger_tip"]["x"] * scale_x)
                hand_y = int(hand["index_finger_tip"]["y"] * scale_y)

                # Draw pointer
                cv2.circle(img, (hand_x, hand_y), 15, (255, 255, 255), -1)
                cv2.circle(img, (hand_x, hand_y), 15, (0, 255, 0), 2)

                # Show pinch status
                if "landmarks" in hand and len(hand["landmarks"]) > 8:
                    thumb_tip = hand["landmarks"][4]
                    index_tip = hand["landmarks"][8]
                    distance = ((thumb_tip["x"] - index_tip["x"]) ** 2 +
                                (thumb_tip["y"] - index_tip["y"]) ** 2) ** 0.5

                    if distance < 30:  # Pinching
                        cv2.circle(img, (hand_x, hand_y), 20, (0, 255, 0), 3)

    def _draw_hud(self, img):
        """Draw the heads-up display"""
        # Timer
        cv2.putText(img, f"Time: {self.time_remaining}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Score
        cv2.putText(img, f"Score: {self.score}", (GAME_WIDTH - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Level and progress
        if self.selected_level is not None and 0 < self.selected_level <= len(LEVELS):
            level_name = LEVELS[self.selected_level - 1]["name"]
            cv2.putText(img, f"Level {self.selected_level}: {level_name}", (GAME_WIDTH // 2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Progress
            cv2.putText(img, f"Pieces: {self.pieces_placed}/{self.total_pieces}", (GAME_WIDTH // 2 - 60, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_level_selection(self, img):
        """Draw level selection screen"""
        cv2.putText(img, "SELECT LEVEL", (GAME_WIDTH // 2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Draw level buttons
        for i, level in enumerate(LEVELS):
            row = i // 3
            col = i % 3
            x = 150 + col * 200
            y = 200 + row * 100

            # Draw button
            cv2.rectangle(img, (x, y), (x + 150, y + 60), (100, 100, 255), -1)
            cv2.rectangle(img, (x, y), (x + 150, y + 60), (255, 255, 255), 2)

            # Draw text
            cv2.putText(img, f"Level {level['level']}", (x + 20, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, level['name'], (x + 20, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _draw_game_over(self, img):
        """Draw game over screen"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Show different message based on completion
        if self.pieces_placed >= self.total_pieces:
            cv2.putText(img, "LEVEL COMPLETE!", (GAME_WIDTH // 2 - 200, GAME_HEIGHT // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "TIME'S UP!", (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)