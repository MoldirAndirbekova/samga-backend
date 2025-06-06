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
PREVIEW_DURATION = 3
SNAP_THRESHOLD = 40
PLACEMENT_SCORE = 10
TIME_BONUS_MULTIPLIER = 0.5

# Level configurations
LEVELS = [
    {"level": 1, "name": "Simple Shapes", "duration": 90, "pieces": 4, "theme": "basic"},
    {"level": 2, "name": "Flower Garden", "duration": 100, "pieces": 5, "theme": "nature"},
    {"level": 3, "name": "Racing Car", "duration": 110, "pieces": 6, "theme": "vehicle"},
    {"level": 4, "name": "Magic Tree", "duration": 120, "pieces": 7, "theme": "nature"},
    {"level": 5, "name": "Express Train", "duration": 130, "pieces": 8, "theme": "vehicle"},
    {"level": 6, "name": "Rainbow Bridge", "duration": 140, "pieces": 7, "theme": "colorful"},
    {"level": 7, "name": "Dream House", "duration": 150, "pieces": 9, "theme": "building"},
    {"level": 8, "name": "Wiggle Worm", "duration": 160, "pieces": 10, "theme": "animal"},
    {"level": 9, "name": "Royal Castle", "duration": 180, "pieces": 12, "theme": "building"}
]

# Shape definitions for each theme
THEME_SHAPES = {
    "basic": ["circle", "rectangle", "triangle", "diamond"],
    "nature": ["circle", "oval", "triangle", "leaf", "flower"],
    "vehicle": ["rectangle", "circle", "triangle", "diamond", "hexagon"],
    "colorful": ["star", "heart", "diamond", "circle", "triangle"],
    "building": ["rectangle", "triangle", "diamond", "hexagon", "pentagon"],
    "animal": ["oval", "circle", "curved", "triangle", "diamond"]
}

# Color schemes for each theme
THEME_COLORS = {
    "basic": [(100, 150, 255), (255, 100, 100), (100, 255, 100), (255, 255, 100)],
    "nature": [(34, 139, 34), (255, 182, 193), (50, 205, 50), (255, 105, 180), (154, 205, 50)],
    "vehicle": [(255, 0, 0), (0, 0, 255), (128, 128, 128), (255, 165, 0), (0, 255, 255)],
    "colorful": [(255, 0, 255), (255, 215, 0), (0, 191, 255), (255, 69, 0), (138, 43, 226)],
    "building": [(139, 69, 19), (128, 128, 128), (255, 140, 0), (205, 133, 63), (160, 82, 45)],
    "animal": [(255, 182, 193), (255, 160, 122), (222, 184, 135), (250, 128, 114), (255, 218, 185)]
}


class GamePiece:
    def __init__(self, piece_id, start_pos, target_pos, shape_type, color, size=60):
        self.piece_id = piece_id
        self.position = list(start_pos)
        self.target_position = list(target_pos)
        self.shape_type = shape_type
        self.color = color
        self.size = size
        self.is_dragging = False
        self.is_placed_correctly = False
        self.image = None

        # Create the piece image
        self.create_piece_image()

    def create_piece_image(self):
        """Create the visual representation of the piece"""
        canvas_size = self.size + 20
        img = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        center = canvas_size // 2

        # Main shape
        if self.shape_type == "circle":
            cv2.circle(img, (center, center), self.size // 2, (*self.color, 255), -1)
            cv2.circle(img, (center, center), self.size // 2, (255, 255, 255, 255), 3)

        elif self.shape_type == "rectangle":
            cv2.rectangle(img, (center - self.size // 2, center - self.size // 2),
                          (center + self.size // 2, center + self.size // 2), (*self.color, 255), -1)
            cv2.rectangle(img, (center - self.size // 2, center - self.size // 2),
                          (center + self.size // 2, center + self.size // 2), (255, 255, 255, 255), 3)

        elif self.shape_type == "triangle":
            points = np.array([
                [center, center - self.size // 2],
                [center - self.size // 2, center + self.size // 2],
                [center + self.size // 2, center + self.size // 2]
            ], np.int32)
            cv2.fillPoly(img, [points], (*self.color, 255))
            cv2.polylines(img, [points], True, (255, 255, 255, 255), 3)

        elif self.shape_type == "diamond":
            points = np.array([
                [center, center - self.size // 2],
                [center + self.size // 2, center],
                [center, center + self.size // 2],
                [center - self.size // 2, center]
            ], np.int32)
            cv2.fillPoly(img, [points], (*self.color, 255))
            cv2.polylines(img, [points], True, (255, 255, 255, 255), 3)

        elif self.shape_type == "star":
            points = self._create_star_points(center, self.size // 2)
            cv2.fillPoly(img, [points], (*self.color, 255))
            cv2.polylines(img, [points], True, (255, 255, 255, 255), 3)

        elif self.shape_type == "heart":
            self._draw_heart(img, center, self.size // 2, self.color)

        elif self.shape_type == "oval":
            cv2.ellipse(img, (center, center), (self.size // 2, self.size // 3), 0, 0, 360, (*self.color, 255), -1)
            cv2.ellipse(img, (center, center), (self.size // 2, self.size // 3), 0, 0, 360, (255, 255, 255, 255), 3)

        elif self.shape_type == "hexagon":
            points = self._create_hexagon_points(center, self.size // 2)
            cv2.fillPoly(img, [points], (*self.color, 255))
            cv2.polylines(img, [points], True, (255, 255, 255, 255), 3)

        elif self.shape_type == "pentagon":
            points = self._create_pentagon_points(center, self.size // 2)
            cv2.fillPoly(img, [points], (*self.color, 255))
            cv2.polylines(img, [points], True, (255, 255, 255, 255), 3)

        else:  # default to circle
            cv2.circle(img, (center, center), self.size // 2, (*self.color, 255), -1)
            cv2.circle(img, (center, center), self.size // 2, (255, 255, 255, 255), 3)

        # Add piece number
        cv2.putText(img, str(self.piece_id + 1), (center - 8, center + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, 255), 2)

        self.image = img

    def _create_star_points(self, center, radius):
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            r = radius if i % 2 == 0 else radius // 2
            x = center + int(r * math.cos(angle - math.pi / 2))
            y = center + int(r * math.sin(angle - math.pi / 2))
            points.append([x, y])
        return np.array(points, np.int32)

    def _create_hexagon_points(self, center, radius):
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center + int(radius * math.cos(angle))
            y = center + int(radius * math.sin(angle))
            points.append([x, y])
        return np.array(points, np.int32)

    def _create_pentagon_points(self, center, radius):
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            x = center + int(radius * math.cos(angle))
            y = center + int(radius * math.sin(angle))
            points.append([x, y])
        return np.array(points, np.int32)

    def _draw_heart(self, img, center, size, color):
        # Simple heart shape using two circles and a triangle
        cv2.circle(img, (center - size // 3, center - size // 3), size // 3, (*color, 255), -1)
        cv2.circle(img, (center + size // 3, center - size // 3), size // 3, (*color, 255), -1)
        points = np.array([
            [center - size // 2, center - size // 6],
            [center + size // 2, center - size // 6],
            [center, center + size // 2]
        ], np.int32)
        cv2.fillPoly(img, [points], (*color, 255))

    def update(self, hand_position, is_pinching):
        """Update piece position based on hand interaction"""
        if self.is_placed_correctly:
            return False

        x, y = self.position
        size = self.size + 20

        # Check if hand is over the piece
        if x < hand_position[0] < x + size and y < hand_position[1] < y + size:
            if is_pinching:
                self.is_dragging = True

        # Update position if dragging
        if self.is_dragging:
            self.position = [hand_position[0] - size // 2, hand_position[1] - size // 2]

        # Stop dragging when pinch is released
        if not is_pinching:
            if self.is_dragging:
                # Check if piece is near its target position
                distance = math.sqrt(
                    (self.position[0] - self.target_position[0]) ** 2 +
                    (self.position[1] - self.target_position[1]) ** 2
                )

                if distance < SNAP_THRESHOLD:
                    # Snap to target position
                    self.position = list(self.target_position)
                    self.is_placed_correctly = True
                    return True

            self.is_dragging = False

        return False


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
        self.pieces = []
        self.target_positions = []
        self.score = 0
        self.pieces_placed = 0
        self.total_pieces = 0
        self.showing_preview = False
        self.preview_start_time = None
        self.game_started = False

        # Hand tracking
        self.left_hand = None
        self.right_hand = None

        # Camera frame storage
        self.current_camera_frame = None

        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps

        print(f"Constructor game initialized with ID: {game_id}")

    def start_game(self, level=None):
        """Start the game with specified level"""
        # Set the level
        if level is not None:
            self.selected_level = level
        else:
            self.selected_level = 1

        # Validate level
        if self.selected_level < 1 or self.selected_level > len(LEVELS):
            self.selected_level = 1

        print(f"🎮 STARTING CONSTRUCTOR LEVEL {self.selected_level}")

        # Complete reset
        self.score = 0
        self.pieces_placed = 0
        self.pieces = []
        self.target_positions = []
        self.game_over = False
        self.game_active = False

        # Load level
        self.load_level(self.selected_level)

        # Start preview
        self.showing_preview = True
        self.preview_start_time = datetime.now()
        self.game_started = True

        print(f"✅ Level {self.selected_level} loaded with {len(self.pieces)} pieces")

    def load_level(self, level_num):
        """Load the specified level"""
        level_data = LEVELS[level_num - 1]

        self.total_pieces = level_data["pieces"]
        self.time_remaining = level_data["duration"]
        theme = level_data["theme"]

        print(f"📋 Loading {level_data['name']} (Theme: {theme})")

        # Create target positions based on level
        self.create_target_positions(level_num, self.total_pieces)

        # Create pieces for this level
        self.create_level_pieces(level_num, theme, self.total_pieces)

    def create_target_positions(self, level_num, num_pieces):
        """Create target positions based on level pattern"""
        self.target_positions = []
        center_x = GAME_WIDTH // 2
        center_y = GAME_HEIGHT // 2

        if level_num == 1:  # Simple 2x2 grid
            spacing = 120
            for i in range(min(4, num_pieces)):
                x = center_x + (i % 2 - 0.5) * spacing
                y = center_y + (i // 2 - 0.5) * spacing
                self.target_positions.append([x, y])

        elif level_num == 2:  # Flower pattern
            self.target_positions.append([center_x, center_y])  # Center
            radius = 100
            for i in range(num_pieces - 1):
                angle = i * 2 * math.pi / (num_pieces - 1)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                self.target_positions.append([x, y])

        elif level_num == 3:  # Car (linear)
            spacing = 100
            start_x = center_x - (num_pieces - 1) * spacing // 2
            for i in range(num_pieces):
                self.target_positions.append([start_x + i * spacing, center_y])

        elif level_num == 4:  # Tree pattern
            # Trunk
            self.target_positions.extend([
                [center_x, center_y + 100],  # Base
                [center_x, center_y + 50],  # Middle trunk
                [center_x, center_y]  # Top trunk
            ])
            # Branches
            self.target_positions.extend([
                [center_x - 80, center_y - 50],  # Left branch
                [center_x + 80, center_y - 50],  # Right branch
                [center_x - 40, center_y - 100],  # Left top
                [center_x + 40, center_y - 100]  # Right top
            ])

        elif level_num == 5:  # Train (linear with slight offset)
            spacing = 90
            start_x = center_x - (num_pieces - 1) * spacing // 2
            for i in range(num_pieces):
                y_offset = 20 * math.sin(i * 0.5) if i > 0 else 0
                self.target_positions.append([start_x + i * spacing, center_y + y_offset])

        elif level_num == 6:  # Rainbow arc
            radius = 150
            start_angle = math.pi
            for i in range(num_pieces):
                angle = start_angle - i * math.pi / (num_pieces - 1)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle) + 50
                self.target_positions.append([x, y])

        elif level_num == 7:  # House
            positions = [
                [center_x - 100, center_y + 80],  # Foundation
                [center_x, center_y + 80],
                [center_x + 100, center_y + 80],
                [center_x - 100, center_y + 20],  # Walls
                [center_x + 100, center_y + 20],
                [center_x - 60, center_y - 40],  # Roof
                [center_x, center_y - 60],
                [center_x + 60, center_y - 40],
                [center_x, center_y + 50]  # Door
            ]
            self.target_positions = positions[:num_pieces]

        elif level_num == 8:  # Worm (wavy)
            amplitude = 80
            frequency = 0.8
            spacing = 70
            start_x = center_x - (num_pieces - 1) * spacing // 2
            for i in range(num_pieces):
                x = start_x + i * spacing
                y = center_y + amplitude * math.sin(frequency * i)
                self.target_positions.append([x, y])

        elif level_num == 9:  # Castle (complex)
            positions = [
                # Base
                [center_x - 150, center_y + 100],
                [center_x - 75, center_y + 100],
                [center_x, center_y + 100],
                [center_x + 75, center_y + 100],
                [center_x + 150, center_y + 100],
                # Middle
                [center_x - 150, center_y + 20],
                [center_x - 50, center_y + 40],
                [center_x + 50, center_y + 40],
                [center_x + 150, center_y + 20],
                # Towers
                [center_x - 150, center_y - 60],
                [center_x, center_y - 80],
                [center_x + 150, center_y - 60]
            ]
            self.target_positions = positions[:num_pieces]

    def create_level_pieces(self, level_num, theme, num_pieces):
        """Create pieces for the level"""
        self.pieces = []

        # Get theme-specific shapes and colors
        shapes = THEME_SHAPES.get(theme, THEME_SHAPES["basic"])
        colors = THEME_COLORS.get(theme, THEME_COLORS["basic"])

        print(f"🎨 Creating {num_pieces} pieces with theme '{theme}'")

        for i in range(num_pieces):
            # Random start position (not overlapping with targets)
            while True:
                start_x = random.randint(50, GAME_WIDTH - 100)
                start_y = random.randint(50, GAME_HEIGHT - 100)

                # Check distance from all target positions
                too_close = False
                for target in self.target_positions:
                    dist = math.sqrt((start_x - target[0]) ** 2 + (start_y - target[1]) ** 2)
                    if dist < 80:  # Minimum distance from targets
                        too_close = True
                        break

                if not too_close:
                    break

            # Select shape and color for this piece
            shape = shapes[i % len(shapes)]
            color = colors[i % len(colors)]
            target_pos = self.target_positions[i]

            # Create piece
            piece = GamePiece(i, [start_x, start_y], target_pos, shape, color)
            self.pieces.append(piece)

            print(f"  Piece {i + 1}: {shape} at ({start_x}, {start_y}) -> ({target_pos[0]}, {target_pos[1]})")

    def update_hands(self, left_hand, right_hand):
        """Update hand positions"""
        self.left_hand = left_hand
        self.right_hand = right_hand

    def update_camera_frame(self, frame):
        """Store camera frame"""
        self.current_camera_frame = frame

    def check_hand_interactions(self):
        """Check for hand interactions with pieces"""
        if not self.pieces:
            return

        # Scale hand coordinates
        scale_x = GAME_WIDTH / 640
        scale_y = GAME_HEIGHT / 480

        for hand in [self.left_hand, self.right_hand]:
            if hand and "index_finger_tip" in hand and "landmarks" in hand:
                hand_x = int(hand["index_finger_tip"]["x"] * scale_x)
                hand_y = int(hand["index_finger_tip"]["y"] * scale_y)

                # Check pinch gesture
                if len(hand["landmarks"]) > 8:
                    thumb_tip = hand["landmarks"][4]
                    index_tip = hand["landmarks"][8]
                    distance = ((thumb_tip["x"] - index_tip["x"]) ** 2 +
                                (thumb_tip["y"] - index_tip["y"]) ** 2) ** 0.5
                    is_pinching = distance < 30

                    # Update pieces
                    for piece in self.pieces:
                        if piece.update((hand_x, hand_y), is_pinching):
                            self.pieces_placed += 1
                            self.score += PLACEMENT_SCORE
                            print(
                                f"🎯 Piece {piece.piece_id + 1} placed! Progress: {self.pieces_placed}/{self.total_pieces}")

    def update_game_state(self):
        """Update game state"""
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
                print(f"🎬 Preview ended, starting level {self.selected_level}")
            return

        if not self.game_active or self.game_over:
            return

        self.last_update = now

        # Update timer
        elapsed_seconds = (now - self.start_time).total_seconds()
        self.time_remaining = max(0, LEVELS[self.selected_level - 1]["duration"] - int(elapsed_seconds))

        # Check win condition
        if self.pieces_placed >= self.total_pieces:
            self.game_over = True
            self.game_active = False
            time_bonus = int(self.time_remaining * TIME_BONUS_MULTIPLIER)
            self.score += time_bonus
            print(f"🏆 Level {self.selected_level} completed! Bonus: {time_bonus}, Final score: {self.score}")
            self.save_game_result()
            return

        # Check time up
        if self.time_remaining <= 0:
            self.game_over = True
            self.game_active = False
            print(f"⏰ Time's up! Level {self.selected_level} - Score: {self.score}")
            self.save_game_result()
            return

        # Check interactions
        self.check_hand_interactions()

    def save_game_result(self):
        """Save game result"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            skill_metrics = self.calculate_skill_metrics()

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

            from .games import game_results
            game_results.append(result)

            import asyncio
            asyncio.create_task(self._persist_to_database(result, skill_metrics))

            if len(game_results) > 100:
                game_results.pop(0)

    def calculate_skill_metrics(self):
        """Calculate skill metrics"""
        completion_rate = (self.pieces_placed / self.total_pieces) * 100 if self.total_pieces > 0 else 0
        time_efficiency = (self.time_remaining / LEVELS[self.selected_level - 1]["duration"]) * 100

        return {
            "hand_eye_coordination": min(100, completion_rate),
            "spatial_reasoning": min(100, (completion_rate + time_efficiency) / 2),
            "memory": min(100, completion_rate * 0.9),
            "creativity": min(100,
                              self.score / (self.total_pieces * PLACEMENT_SCORE) * 100) if self.total_pieces > 0 else 0
        }

    async def _persist_to_database(self, result, skill_metrics):
        """Save to database"""
        try:
            from database import prisma
            if not prisma.is_connected():
                await prisma.connect()

            await prisma.gamereport.create(
                data={
                    "gameId": result["game_id"],
                    "gameTypeId": "constructor",
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
        except Exception as e:
            print(f"Database save error: {e}")

    def render_frame(self):
        """Render game frame"""
        # Create background
        if self.current_camera_frame is not None:
            try:
                if isinstance(self.current_camera_frame, str):
                    image_data = base64.b64decode(
                        self.current_camera_frame.split(',')[1]
                        if ',' in self.current_camera_frame
                        else self.current_camera_frame
                    )
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    img = self.current_camera_frame

                if img is not None:
                    img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))
                else:
                    img = self._create_background()
            except:
                img = self._create_background()
        else:
            img = self._create_background()

        # Add overlay
        overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        overlay.fill(20)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Show preview
        if self.showing_preview:
            self._draw_preview(img)
        # Show active game
        elif self.game_active:
            self._draw_game(img)
        # Show level selection
        elif not self.game_started:
            self._draw_level_selection(img)

        # Show game over
        if self.game_over:
            self._draw_game_over(img)

        # Convert to base64
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        return None

    def _create_background(self):
        """Create default background"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        for y in range(GAME_HEIGHT):
            color = int(80 + (y / GAME_HEIGHT) * 100)
            cv2.line(img, (0, y), (GAME_WIDTH, y), (color, color, color + 50), 1)
        return img

    def _draw_preview(self, img):
        """Draw preview screen"""
        level_data = LEVELS[self.selected_level - 1]

        # Title
        cv2.putText(img, f"LEVEL {self.selected_level}",
                    (GAME_WIDTH // 2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Level name
        cv2.putText(img, level_data["name"],
                    (GAME_WIDTH // 2 - len(level_data["name"]) * 10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 100), 2)

        # Instructions
        cv2.putText(img, "STUDY THE PATTERN",
                    (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(img, f"Place {self.total_pieces} pieces in position",
                    (GAME_WIDTH // 2 - 130, GAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Countdown
        time_left = PREVIEW_DURATION - (datetime.now() - self.preview_start_time).total_seconds()
        cv2.putText(img, f"Starting in: {int(time_left + 1)}s",
                    (GAME_WIDTH // 2 - 80, GAME_HEIGHT - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    def _draw_game(self, img):
        """Draw active game"""
        # Draw target positions
        for i, target in enumerate(self.target_positions):
            if i < len(self.pieces) and not self.pieces[i].is_placed_correctly:
                cv2.circle(img, (int(target[0]), int(target[1])), 30, (255, 255, 255), 2)
                cv2.putText(img, str(i + 1),
                            (int(target[0]) - 8, int(target[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw pieces
        for piece in self.pieces:
            if piece.image is not None:
                self._overlay_image(img, piece.image, piece.position)

                if piece.is_placed_correctly:
                    # Green checkmark
                    x, y = piece.position
                    cv2.circle(img, (int(x + 40), int(y + 40)), 15, (0, 255, 0), -1)
                    cv2.putText(img, "✓", (int(x + 35), int(y + 45)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw hand indicators
        self._draw_hands(img)

        # Draw HUD
        self._draw_hud(img)

    def _draw_hands(self, img):
        """Draw hand indicators"""
        scale_x = GAME_WIDTH / 640
        scale_y = GAME_HEIGHT / 480

        for hand in [self.left_hand, self.right_hand]:
            if hand and "index_finger_tip" in hand:
                x = int(hand["index_finger_tip"]["x"] * scale_x)
                y = int(hand["index_finger_tip"]["y"] * scale_y)

                cv2.circle(img, (x, y), 15, (255, 255, 255), -1)
                cv2.circle(img, (x, y), 15, (0, 255, 0), 2)

                # Show pinch
                if "landmarks" in hand and len(hand["landmarks"]) > 8:
                    thumb = hand["landmarks"][4]
                    index = hand["landmarks"][8]
                    distance = ((thumb["x"] - index["x"]) ** 2 + (thumb["y"] - index["y"]) ** 2) ** 0.5
                    if distance < 30:
                        cv2.circle(img, (x, y), 20, (255, 0, 0), 3)

    def _draw_hud(self, img):
        """Draw HUD"""
        # Timer
        cv2.putText(img, f"Time: {self.time_remaining}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Score
        cv2.putText(img, f"Score: {self.score}", (GAME_WIDTH - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Level
        level_name = LEVELS[self.selected_level - 1]["name"]
        cv2.putText(img, f"Level {self.selected_level}: {level_name}",
                    (GAME_WIDTH // 2 - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Progress
        cv2.putText(img, f"Pieces: {self.pieces_placed}/{self.total_pieces}",
                    (GAME_WIDTH // 2 - 60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_level_selection(self, img):
        """Draw level selection"""
        cv2.putText(img, "SELECT LEVEL", (GAME_WIDTH // 2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def _draw_game_over(self, img):
        """Draw game over screen"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        if self.pieces_placed >= self.total_pieces:
            cv2.putText(img, "LEVEL COMPLETE!", (GAME_WIDTH // 2 - 200, GAME_HEIGHT // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "TIME'S UP!", (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def _overlay_image(self, background, overlay, position):
        """Overlay image with alpha"""
        x, y = int(position[0]), int(position[1])
        h, w = overlay.shape[:2]

        # Bounds check
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

        if overlay.shape[2] == 4:  # Has alpha
            alpha = overlay[:h, :w, 3] / 255.0
            for c in range(3):
                background[y:y + h, x:x + w, c] = (
                        alpha * overlay[:h, :w, c] +
                        (1 - alpha) * background[y:y + h, x:x + w, c]
                )
        else:
            background[y:y + h, x:x + w] = overlay[:h, :w, :3]