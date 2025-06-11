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
PREVIEW_DURATION = 10  # 10 seconds like in the original ConstructorWithGames.py
SNAP_THRESHOLD = 40
PLACEMENT_SCORE = 10
TIME_BONUS_MULTIPLIER = 0.5

# Level configurations - using the same structure as ConstructorWithGames.py
LEVELS = [
    {"level": 1, "folder": "mushroom", "duration": 120, "name": "Magic Mushroom"},
    {"level": 2, "folder": "flower", "duration": 120, "name": "Beautiful Flower"},
    {"level": 3, "folder": "car", "duration": 140, "name": "Racing Car"},
    {"level": 4, "folder": "tree", "duration": 140, "name": "Magic Tree"},
    {"level": 5, "folder": "train", "duration": 140, "name": "Express Train"},
    {"level": 6, "folder": "rainbow", "duration": 200, "name": "Rainbow Bridge"},
    {"level": 7, "folder": "home", "duration": 220, "name": "Dream House"},
    {"level": 8, "folder": "worm", "duration": 220, "name": "Wiggle Worm"},
    {"level": 9, "folder": "castle", "duration": 240, "name": "Royal Castle"}
]

# Base path for constructor assets
CONSTRUCTOR_ASSETS_PATH = os.path.join("game_assets", "constructor")


class GamePiece:
    def __init__(self, piece_id, start_pos, target_pos, image_path):
        self.piece_id = piece_id
        self.position = list(start_pos)
        self.target_position = list(target_pos)
        self.image_path = image_path
        self.is_dragging = False
        self.is_placed_correctly = False
        self.has_been_picked_up = False  # Track if piece has been picked up
        self.pickup_count = 0  # Count how many times piece was picked up
        self.image = None
        self.size = (80, 80)  # Default size

        # Load the piece image
        self.load_piece_image()

    def load_piece_image(self):
        """Load the actual piece image from file WITHOUT background removal"""
        # Always create fallback first to ensure we have something
        self._create_fallback_image()

        try:
            if os.path.exists(self.image_path):
                print(f"Loading piece image: {self.image_path}")
                loaded_img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)

                if loaded_img is not None and loaded_img.size > 0:
                    # Convert to RGBA if needed
                    if len(loaded_img.shape) == 3:
                        loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2BGRA)

                    # Resize to consistent size but keep original appearance
                    scale_factor = 0.4  # Smaller for better performance
                    processed_img = cv2.resize(loaded_img, (0, 0), fx=scale_factor, fy=scale_factor)

                    # Only replace fallback if processing was successful
                    if processed_img.size > 0:
                        self.image = processed_img
                        self.size = processed_img.shape[:2]

                        print(f"✅ Loaded original piece {self.piece_id + 1} from {self.image_path}")
                    else:
                        print(f"⚠️ Processed image is empty for {self.image_path}, using fallback")
                else:
                    print(f"⚠️ Could not load image data from {self.image_path}, using fallback")
            else:
                print(f"⚠️ Image file not found: {self.image_path}, using fallback")
        except Exception as e:
            print(f"⚠️ Error loading piece image {self.image_path}: {e}, using fallback")

    def _create_fallback_image(self):
        """Create a fallback image if the actual image can't be loaded"""
        canvas_size = 60  # Smaller for better performance
        self.image = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        center = canvas_size // 2

        # Create a colored rectangle as fallback
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
            (255, 100, 255), (100, 255, 255), (255, 150, 100), (150, 255, 100),
            (100, 150, 255), (255, 200, 100)
        ]
        color = colors[self.piece_id % len(colors)]

        # Create solid rectangle with border
        cv2.rectangle(self.image, (5, 5), (canvas_size - 5, canvas_size - 5), (*color, 255), -1)
        cv2.rectangle(self.image, (5, 5), (canvas_size - 5, canvas_size - 5), (255, 255, 255, 255), 2)

        # Add piece number with better visibility
        number_text = str(self.piece_id + 1)
        text_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center - text_size[0] // 2
        text_y = center + text_size[1] // 2

        # White number
        cv2.putText(self.image, number_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)

        self.size = (canvas_size, canvas_size)
        print(f"✅ Created fallback image for piece {self.piece_id + 1}")

    def update(self, hand_position, is_pinching):
        """Update piece position based on hand interaction with boundary checking - optimized"""
        if self.is_placed_correctly:
            return False

        x, y = self.position
        h, w = self.size

        # Optimized collision detection
        if x < hand_position[0] < x + w and y < hand_position[1] < y + h:
            if is_pinching and not self.is_dragging:
                self.is_dragging = True

        # Update position if dragging - optimized bounds checking
        if self.is_dragging:
            new_x = hand_position[0] - w // 2
            new_y = hand_position[1] - h // 2

            # Simplified bounds checking
            new_x = max(5, min(new_x, GAME_WIDTH - w - 5))
            new_y = max(80, min(new_y, GAME_HEIGHT - h - 5))

            self.position = [new_x, new_y]

        # Stop dragging when pinch is released
        if not is_pinching and self.is_dragging:
            # Optimized snap detection
            piece_center_x = self.position[0] + w // 2
            piece_center_y = self.position[1] + h // 2

            # Simple distance check
            dx = piece_center_x - self.target_position[0]
            dy = piece_center_y - self.target_position[1]
            distance_sq = dx * dx + dy * dy  # Skip sqrt for performance

            if distance_sq < (SNAP_THRESHOLD * SNAP_THRESHOLD):
                # Snap to target position
                self.position = [
                    self.target_position[0] - w // 2,
                    self.target_position[1] - h // 2
                ]
                self.is_placed_correctly = True
                self.is_dragging = False
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
        self.preview_image = None

        # Scoring tracking
        self.total_pickup_actions = 0
        self.total_placement_actions = 0
        self.perfect_order_placements = 0

        # Hand tracking
        self.left_hand = None
        self.right_hand = None

        # Camera frame storage
        self.current_camera_frame = None

        # FPS control - optimized for better performance
        self.fps = 30  # Reduced from 60 for better performance
        self.frame_time = 1 / self.fps

        print(f"Constructor game initialized with ID: {game_id}")

    def get_small_reference_image(self):
        """Get a small version of the preview image for reference during gameplay"""
        try:
            if hasattr(self, 'preview_image') and self.preview_image is not None:
                # Create a smaller reference image for better performance
                small_ref = cv2.resize(self.preview_image, (150, 112))  # Smaller but proportional

                # Add a thin border
                bordered = cv2.copyMakeBorder(small_ref, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                return bordered
            else:
                print("⚠️ No preview image available for reference")
                # Return a smaller placeholder image
                placeholder = np.zeros((116, 154, 3), dtype=np.uint8)
                placeholder.fill(50)
                cv2.putText(placeholder, "No Preview", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return placeholder
        except Exception as e:
            print(f"❌ Error creating small reference image: {e}")
            # Return a simple placeholder
            placeholder = np.zeros((116, 154, 3), dtype=np.uint8)
            placeholder.fill(100)
            cv2.putText(placeholder, "Error", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return placeholder

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
        self.preview_image = None

        # Reset scoring tracking
        self.total_pickup_actions = 0
        self.total_placement_actions = 0
        self.perfect_order_placements = 0

        # Load level
        try:
            self.load_level(self.selected_level)
        except Exception as e:
            print(f"❌ Error loading level: {e}")

        # Start preview
        self.showing_preview = True
        self.preview_start_time = datetime.now()
        self.game_started = True

        print(f"✅ Level {self.selected_level} loaded with {len(self.pieces)} pieces")

    def load_level(self, level_num):
        """Load the specified level with real images"""
        level_data = LEVELS[level_num - 1]
        folder_name = level_data["folder"]

        self.time_remaining = level_data["duration"]
        level_folder = os.path.join(CONSTRUCTOR_ASSETS_PATH, folder_name)

        print(f"📋 Loading {level_data['name']} from folder: {level_folder}")

        # Load preview image
        self.load_preview_image(level_folder, level_num)

        # Get all piece files (excluding level preview)
        piece_files = self.get_piece_files(level_folder)
        self.total_pieces = len(piece_files)

        print(f"Found {self.total_pieces} pieces in {level_folder}")

        if self.total_pieces == 0:
            print(f"⚠️ No pieces found in {level_folder}, creating fallback pieces")
            self.total_pieces = 4  # Fallback
            piece_files = [f"{i + 1}.png" for i in range(self.total_pieces)]

        # Create target positions based on level pattern
        self.create_target_positions(level_num, self.total_pieces)

        # Create pieces from actual image files
        self.create_level_pieces(level_folder, piece_files)

    def load_preview_image(self, level_folder, level_num):
        """Load the preview image for the level"""
        preview_path = os.path.join(level_folder, f"level{level_num}.png")

        if os.path.exists(preview_path):
            try:
                self.preview_image = cv2.imread(preview_path)
                if self.preview_image is not None:
                    # Resize to fit game window
                    self.preview_image = cv2.resize(self.preview_image, (GAME_WIDTH, GAME_HEIGHT))
                    print(f"✅ Loaded preview image: {preview_path}")
                else:
                    print(f"❌ Failed to load preview image: {preview_path}")
            except Exception as e:
                print(f"❌ Error loading preview image: {e}")
        else:
            print(f"⚠️ Preview image not found: {preview_path}")

    def get_piece_files(self, level_folder):
        """Get all piece files from the level folder"""
        piece_files = []

        if os.path.exists(level_folder):
            all_files = os.listdir(level_folder)
            # Filter out the level preview file and get only numbered pieces
            for file in all_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.lower().startswith('level'):
                    piece_files.append(file)

            # Sort by number if possible
            try:
                piece_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except:
                piece_files.sort()

        return piece_files

    def create_target_positions(self, level_num, num_pieces):
        """Create target positions based on level pattern with safe areas"""
        self.target_positions = []

        # Simplified safe play area
        center_x = GAME_WIDTH // 2 - 50  # Shift left to avoid reference image
        center_y = GAME_HEIGHT // 2

        # Simple grid layout for all levels to ensure visibility
        if num_pieces <= 4:
            # 2x2 grid for small numbers
            spacing = 120
            positions = [
                [center_x - spacing // 2, center_y - spacing // 2],  # Top-left
                [center_x + spacing // 2, center_y - spacing // 2],  # Top-right
                [center_x - spacing // 2, center_y + spacing // 2],  # Bottom-left
                [center_x + spacing // 2, center_y + spacing // 2],  # Bottom-right
            ]
            for i in range(min(num_pieces, 4)):
                self.target_positions.append(positions[i])

        else:
            # Larger grid for more pieces
            cols = int(math.ceil(math.sqrt(num_pieces)))
            spacing = 100

            start_x = center_x - (cols - 1) * spacing // 2
            start_y = center_y - ((num_pieces + cols - 1) // cols - 1) * spacing // 2

            for i in range(num_pieces):
                row = i // cols
                col = i % cols
                x = start_x + col * spacing
                y = start_y + row * spacing

                # Keep within reasonable bounds
                x = max(100, min(x, GAME_WIDTH - 300))  # Leave space for reference
                y = max(200, min(y, GAME_HEIGHT - 100))  # Leave space for HUD

                self.target_positions.append([x, y])

    def create_level_pieces(self, level_folder, piece_files):
        """Create pieces from actual image files with proper positioning"""
        self.pieces = []

        print(f"🎨 Creating {len(piece_files)} pieces from folder: {level_folder}")
        print(f"🎨 Piece files: {piece_files}")
        print(f"🎨 Target positions: {len(self.target_positions)}")

        for i, piece_file in enumerate(piece_files):
            # Simple spawn positioning to avoid complexity
            start_x = 50 + (i % 4) * 150  # Spread horizontally
            start_y = 100 + (i // 4) * 100  # Stack vertically if needed

            # Ensure position is within bounds
            start_x = max(20, min(start_x, GAME_WIDTH - 100))
            start_y = max(100, min(start_y, GAME_HEIGHT - 100))

            # Get target position - ensure piece number matches target number
            if i < len(self.target_positions):
                target_pos = self.target_positions[i]
            else:
                target_pos = [GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2]

            # Full path to piece image
            image_path = os.path.join(level_folder, piece_file)

            # Create piece
            print(f"🔧 Creating piece {i + 1}: {piece_file}")
            piece = GamePiece(i, [start_x, start_y], target_pos, image_path)
            self.pieces.append(piece)

            print(f"✅ Piece {i + 1}: {piece_file} at ({start_x}, {start_y}) -> ({target_pos[0]}, {target_pos[1]})")

        print(f"🎮 Total pieces created: {len(self.pieces)}")

        # Verify all pieces have images
        for i, piece in enumerate(self.pieces):
            if piece.image is not None:
                print(f"  ✅ Piece {i + 1} has image (size: {piece.size})")
            else:
                print(f"  ❌ Piece {i + 1} has NO image!")
                # Emergency fallback
                piece._create_fallback_image()

    def update_hands(self, left_hand, right_hand):
        """Update hand positions"""
        self.left_hand = left_hand
        self.right_hand = right_hand

    def update_camera_frame(self, frame):
        """Store camera frame"""
        self.current_camera_frame = frame

    def check_hand_interactions(self):
        """Check for hand interactions with pieces and award points for actions"""
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

                    # Update pieces and check for scoring
                    for piece in self.pieces:
                        # Check if piece was just picked up (started dragging)
                        was_dragging = piece.is_dragging

                        # Update piece
                        piece_placed = piece.update((hand_x, hand_y), is_pinching)

                        # Award points for picking up a piece (drag action) - only first time
                        if not was_dragging and piece.is_dragging and not piece.has_been_picked_up:
                            pickup_points = 2  # Points for picking up a piece
                            self.score += pickup_points
                            self.total_pickup_actions += 1
                            piece.has_been_picked_up = True
                            piece.pickup_count += 1
                            print(f"🤏 Picked up piece {piece.piece_id + 1}! +{pickup_points} points")

                        # Award points for successfully placing a piece (drop action)
                        if piece_placed:
                            self.pieces_placed += 1
                            self.total_placement_actions += 1
                            placement_points = PLACEMENT_SCORE

                            # Bonus points for placing pieces in correct order (1, 2, 3...)
                            correct_order = all(
                                self.pieces[i].is_placed_correctly
                                for i in range(piece.piece_id)
                            )

                            if correct_order:
                                order_bonus = 5
                                self.score += placement_points + order_bonus
                                self.perfect_order_placements += 1
                                print(
                                    f"🎯 Piece {piece.piece_id + 1} placed in correct order! +{placement_points + order_bonus} points")
                            else:
                                self.score += placement_points
                                print(f"🎯 Piece {piece.piece_id + 1} placed! +{placement_points} points")

                            print(f"📊 Progress: {self.pieces_placed}/{self.total_pieces}, Score: {self.score}")

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

        # Check interactions and handle scoring
        self.check_hand_interactions()

        # Check win condition
        if self.pieces_placed >= self.total_pieces:
            self.game_over = True
            self.game_active = False

            # Award time bonus
            time_bonus = int(self.time_remaining * TIME_BONUS_MULTIPLIER)
            self.score += time_bonus

            # Award completion bonus
            completion_bonus = 50  # Big bonus for completing the level
            self.score += completion_bonus

            print(f"🏆 Level {self.selected_level} completed!")
            print(f"💰 Time bonus: +{time_bonus} points")
            print(f"🎉 Completion bonus: +{completion_bonus} points")
            print(f"🏅 Final score: {self.score}")

            self.save_game_result()
            return

        # Check time up
        if self.time_remaining <= 0:
            self.game_over = True
            self.game_active = False
            print(f"⏰ Time's up! Level {self.selected_level} - Score: {self.score}")
            self.save_game_result()
            return

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
        """Calculate skill metrics based on enhanced scoring system"""
        completion_rate = (self.pieces_placed / self.total_pieces) * 100 if self.total_pieces > 0 else 0
        time_efficiency = (self.time_remaining / LEVELS[self.selected_level - 1]["duration"]) * 100

        # Calculate interaction efficiency (fewer pickups = better)
        total_pickups = sum(piece.pickup_count for piece in self.pieces)
        pickup_efficiency = max(0, 100 - (total_pickups - self.total_pieces) * 10) if self.total_pieces > 0 else 0

        # Calculate order accuracy
        correct_order_count = 0
        for i, piece in enumerate(self.pieces):
            if piece.is_placed_correctly:
                # Check if all previous pieces are also correctly placed
                if all(self.pieces[j].is_placed_correctly for j in range(i)):
                    correct_order_count += 1

        order_accuracy = (correct_order_count / self.total_pieces) * 100 if self.total_pieces > 0 else 0

        return {
            "hand_eye_coordination": min(100, (completion_rate + pickup_efficiency) / 2),
            "spatial_reasoning": min(100, (completion_rate + order_accuracy) / 2),
            "memory": min(100, order_accuracy * 0.9 + time_efficiency * 0.1),
            "creativity": min(100, self.score / (
                        self.total_pieces * (PLACEMENT_SCORE + 7)) * 100) if self.total_pieces > 0 else 0
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
        try:
            # Create background
            if self.showing_preview and hasattr(self, 'preview_image') and self.preview_image is not None:
                # Show the actual preview image during preview phase
                img = self.preview_image.copy()

                # Add preview overlay
                overlay = np.zeros_like(img, dtype=np.uint8)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            elif hasattr(self, 'current_camera_frame') and self.current_camera_frame is not None:
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

            # Add overlay for active game (not during preview)
            if not self.showing_preview:
                overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
                overlay.fill(20)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Show preview
            if self.showing_preview:
                self._draw_preview(img)
            # Show active game
            elif hasattr(self, 'game_active') and self.game_active:
                self._draw_game(img)
            # Show level selection
            elif not hasattr(self, 'game_started') or not self.game_started:
                self._draw_level_selection(img)

            # Show game over
            if hasattr(self, 'game_over') and self.game_over:
                self._draw_game_over(img)

            # Convert to base64
            success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            return None

        except Exception as e:
            print(f"❌ Error in render_frame: {e}")
            # Return a simple error image
            img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(img, "Rendering Error", (GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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

        # Title overlay on preview image
        cv2.putText(img, f"LEVEL {self.selected_level}",
                    (GAME_WIDTH // 2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Level name
        cv2.putText(img, level_data["name"],
                    (GAME_WIDTH // 2 - len(level_data["name"]) * 10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 100), 2)

        # Instructions
        cv2.putText(img, "MEMORIZE THE PATTERN",
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
        """Draw active game with reference image and scoring feedback"""
        # Draw small reference image at top-right
        try:
            ref_img = self.get_small_reference_image()
            if ref_img is not None:
                ref_h, ref_w = ref_img.shape[:2]
                ref_x = GAME_WIDTH - ref_w - 10
                ref_y = 10

                # Overlay the reference image
                if ref_y + ref_h <= GAME_HEIGHT and ref_x + ref_w <= GAME_WIDTH:
                    img[ref_y:ref_y + ref_h, ref_x:ref_x + ref_w] = ref_img

                    # Add small label
                    cv2.putText(img, "Build:", (ref_x, ref_y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        except Exception as e:
            print(f"❌ Error drawing reference image: {e}")

        # NO TARGET CIRCLES - removed the numbered circles with red frames

        # Draw pieces with original colors preserved
        try:
            for piece in self.pieces:
                # Always draw something for each piece, even if image fails
                piece_drawn = False

                if piece.image is not None:
                    try:
                        self._overlay_image(img, piece.image, piece.position)
                        piece_drawn = True
                    except Exception as e:
                        print(f"❌ Error drawing piece {piece.piece_id}: {e}")

                # Fallback: draw a simple shape if image overlay failed
                if not piece_drawn:
                    x, y = int(piece.position[0]), int(piece.position[1])
                    # Draw a colored rectangle with piece number
                    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
                    color = colors[piece.piece_id % len(colors)]

                    cv2.rectangle(img, (x, y), (x + 60, y + 60), color, -1)
                    cv2.rectangle(img, (x, y), (x + 60, y + 60), (255, 255, 255), 2)

                    # Add piece number
                    cv2.putText(img, str(piece.piece_id + 1), (x + 20, y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Draw status indicators
                if piece.is_placed_correctly:
                    # Small green checkmark 
                    x, y = piece.position
                    h, w = piece.size if hasattr(piece, 'size') else (60, 60)
                    center_x = int(x + w // 2)
                    center_y = int(y + h // 2)

                    # Small green circle
                    cv2.circle(img, (center_x, center_y), 15, (0, 255, 0), -1)
                    cv2.circle(img, (center_x, center_y), 15, (255, 255, 255), 2)

                    # Small checkmark
                    cv2.putText(img, "✓", (center_x - 8, center_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                elif piece.is_dragging:
                    # Subtle highlight for dragging piece
                    x, y = piece.position
                    h, w = piece.size if hasattr(piece, 'size') else (60, 60)
                    center_x = int(x + w // 2)
                    center_y = int(y + h // 2)

                    # Thin yellow outline
                    cv2.circle(img, (center_x, center_y), 35, (255, 255, 0), 2)
        except Exception as e:
            print(f"❌ Error drawing pieces: {e}")

        # Draw hand indicators (smaller for performance)
        try:
            self._draw_hands(img)
        except Exception as e:
            print(f"❌ Error drawing hands: {e}")

        # Draw HUD
        try:
            self._draw_hud(img)
        except Exception as e:
            print(f"❌ Error drawing HUD: {e}")

    def _draw_hands(self, img):
        """Draw hand indicators - optimized for performance"""
        scale_x = GAME_WIDTH / 640
        scale_y = GAME_HEIGHT / 480

        for hand in [self.left_hand, self.right_hand]:
            if hand and "index_finger_tip" in hand:
                x = int(hand["index_finger_tip"]["x"] * scale_x)
                y = int(hand["index_finger_tip"]["y"] * scale_y)

                # Simple hand indicator - optimized
                cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
                cv2.circle(img, (x, y), 8, (0, 255, 0), 1)

                # Show pinch with simpler visualization
                if "landmarks" in hand and len(hand["landmarks"]) > 8:
                    thumb = hand["landmarks"][4]
                    index = hand["landmarks"][8]
                    distance = ((thumb["x"] - index["x"]) ** 2 + (thumb["y"] - index["y"]) ** 2) ** 0.5
                    if distance < 30:
                        cv2.circle(img, (x, y), 12, (255, 0, 0), 2)

    def _draw_hud(self, img):
        """Draw HUD with proper spacing for reference image and detailed scoring"""
        # Left side HUD - Timer and Level info
        cv2.putText(img, f"Time: {self.time_remaining}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        level_name = LEVELS[self.selected_level - 1]["name"]
        cv2.putText(img, f"Level {self.selected_level}: {level_name}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Progress and Score (left side, below level info)
        cv2.putText(img, f"Progress: {self.pieces_placed}/{self.total_pieces}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Score with color coding
        score_color = (100, 255, 100) if self.score > 0 else (255, 255, 255)
        cv2.putText(img, f"Score: {self.score}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

    def _draw_level_selection(self, img):
        """Draw level selection"""
        cv2.putText(img, "SELECT LEVEL", (GAME_WIDTH // 2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def _draw_game_over(self, img):
        """Draw game over screen with detailed scoring breakdown"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        if self.pieces_placed >= self.total_pieces:
            cv2.putText(img, "LEVEL COMPLETE!", (GAME_WIDTH // 2 - 200, GAME_HEIGHT // 2 - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Show detailed scoring breakdown
            y_offset = GAME_HEIGHT // 2 - 100
            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH // 2 - 120, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Breakdown
            y_offset += 40
            cv2.putText(img, "Score Breakdown:", (GAME_WIDTH // 2 - 100, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            y_offset += 25
            cv2.putText(img, f"• Pieces placed: {self.pieces_placed} x 10 = {self.pieces_placed * 10}",
                        (GAME_WIDTH // 2 - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            y_offset += 20
            pickup_count = sum(piece.pickup_count for piece in self.pieces)
            cv2.putText(img, f"• Piece pickups: {pickup_count} x 2 = {pickup_count * 2}",
                        (GAME_WIDTH // 2 - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            y_offset += 20
            time_bonus = int(self.time_remaining * TIME_BONUS_MULTIPLIER)
            cv2.putText(img, f"• Time bonus: {time_bonus}",
                        (GAME_WIDTH // 2 - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            y_offset += 20
            cv2.putText(img, f"• Completion bonus: 50",
                        (GAME_WIDTH // 2 - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        else:
            cv2.putText(img, "TIME'S UP!", (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.putText(img, f"Pieces placed: {self.pieces_placed}/{self.total_pieces}",
                        (GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def _overlay_image(self, background, overlay, position):
        """Overlay image with alpha"""
        try:
            x, y = int(position[0]), int(position[1])
            h, w = overlay.shape[:2]

            # Bounds check
            if x >= background.shape[1] or y >= background.shape[0] or x + w <= 0 or y + h <= 0:
                return

            # Clip overlay to fit within background
            overlay_x_start = max(0, -x)
            overlay_y_start = max(0, -y)
            overlay_x_end = min(w, background.shape[1] - x)
            overlay_y_end = min(h, background.shape[0] - y)

            bg_x_start = max(0, x)
            bg_y_start = max(0, y)
            bg_x_end = min(background.shape[1], x + w)
            bg_y_end = min(background.shape[0], y + h)

            # Get the overlapping regions
            overlay_region = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

            if overlay_region.size == 0:
                return

            if len(overlay_region.shape) == 3 and overlay_region.shape[2] == 4:  # Has alpha
                # Alpha blending
                alpha = overlay_region[:, :, 3:4] / 255.0
                overlay_rgb = overlay_region[:, :, :3]
                background_region = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end]

                # Blend
                blended = overlay_rgb * alpha + background_region * (1 - alpha)
                background[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = blended.astype(np.uint8)
            else:
                # Direct copy for non-alpha images
                background[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = overlay_region[:, :, :3]

        except Exception as e:
            print(f"❌ Error in overlay_image: {e}")
            # Fallback: draw a simple circle to show piece location
            try:
                center_x = int(position[0] + 40)
                center_y = int(position[1] + 40)
                cv2.circle(background, (center_x, center_y), 30, (255, 100, 100), -1)
                cv2.circle(background, (center_x, center_y), 30, (255, 255, 255), 2)
            except:
                pass