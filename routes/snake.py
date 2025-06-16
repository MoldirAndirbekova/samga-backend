# games/snake.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import os
import numpy as np
import cv2
import random
import base64
import math

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
INITIAL_SNAKE_LENGTH = 150
SNAKE_GROWTH = 50
SNAKE_THICKNESS = 20
FOOD_SIZE = 50
GAME_DURATION = 120  # 2 minutes

DIFFICULTY_LEVELS = {
    "EASY": {
        "speed_multiplier": 0.8,
        "score_multiplier": 1
    },
    "MEDIUM": {
        "speed_multiplier": 1.0,
        "score_multiplier": 1.5
    },
    "HARD": {
        "speed_multiplier": 1.5,
        "score_multiplier": 2
    }
}

class SnakeGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id
        
        # Game state
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each points
        self.current_length = 0
        self.allowed_length = INITIAL_SNAKE_LENGTH
        self.previous_head = None
        self.food_point = None
        self.score = 0
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        self.time_remaining = GAME_DURATION
        
        # Hand tracking
        self.hand = None
        
        # Camera frame storage for AR overlay
        self.current_camera_frame = None
        
        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps

        # Load multiple food images
        self.food_images = {}
        self.current_food_image = None
        self.load_food_images()
        
        # Initialize food
        self.random_food_location()
    
    def load_food_images(self):
        """Load all food images (1.png through 12.png)"""
        print("Loading food images...")
        
        for i in range(1, 13):  # 1.png through 12.png
            try:
                food_path = f'/app/static/images/{i}.png'
                
                print(f"Attempting to load food image from: {food_path}")
                print(f"File exists: {os.path.exists(food_path)}")
                
                food_image = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)
                
                if food_image is not None:
                    # Resize to desired size
                    food_image = cv2.resize(food_image, (FOOD_SIZE, FOOD_SIZE))
                    self.food_images[i] = food_image
                    print(f"Food image {i}.png loaded successfully: {food_image.shape}")
                else:
                    print(f"OpenCV failed to load image from {food_path}")
                    # Additional debug info
                    if os.path.exists(food_path):
                        print(f"File size: {os.path.getsize(food_path)}")
                    
            except Exception as e:
                print(f"Error loading food image {i}.png: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Successfully loaded {len(self.food_images)} food images")
        
        # If no images loaded, create a fallback
        if not self.food_images:
            print("No food images loaded, will use fallback circles")
    
    def select_random_food_image(self):
        """Select a random food image from the loaded images"""
        if self.food_images:
            # Randomly select one of the loaded food images
            image_key = random.choice(list(self.food_images.keys()))
            self.current_food_image = self.food_images[image_key]
            print(f"Selected food image: {image_key}.png")
        else:
            self.current_food_image = None
            print("No food images available, using fallback")
    
    def random_food_location(self):
        """Generate random location for food and select random food image"""
        self.food_point = (
            random.randint(100, GAME_WIDTH - 100),
            random.randint(100, GAME_HEIGHT - 100)
        )
        # Select a new random food image each time food is placed
        self.select_random_food_image()
    
    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Snake game id: {self.game_id} with difficulty: {self.difficulty}")
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.score = 0
        self.time_remaining = GAME_DURATION
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = INITIAL_SNAKE_LENGTH
        self.previous_head = None
        self.random_food_location()
    
    def update_hand(self, hand):
        """Update hand position based on hand tracking data"""
        self.hand = hand
    
    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame

    def pause_game(self):
        """Pause the game"""
        print(f"Pausing Snake game id: {self.game_id}")
        self.game_active = False
    
    def resume_game(self):
        """Resume the game"""
        print(f"Resuming Snake game id: {self.game_id}")
        self.game_active = True
        self.last_update = datetime.now()  # Reset the last update time to prevent time jumps
    
    def update_game_state(self):
        """Update the game state for one frame"""
        if not self.game_active or self.game_over:
            return
        
        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now
        
        # Don't update time when paused
        if self.game_active:
            # Update game timer
            elapsed_seconds = (now - self.start_time).total_seconds()
            self.time_remaining = max(0, GAME_DURATION - int(elapsed_seconds))
        
    
    def update_game_state(self):
        """Update the game state for one frame"""
        if not self.game_active or self.game_over:
            return
        
        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now
        
        # Update game timer
        elapsed_seconds = (now - self.start_time).total_seconds()
        self.time_remaining = max(0, GAME_DURATION - int(elapsed_seconds))
        
        # Check if game is over
        if self.time_remaining <= 0:
            print("Game over - time's up!")
            self.game_over = True
            self.game_active = False
            self.save_game_result()
            return
        
        # Update snake based on hand position
        if self.hand:
            # Use index finger tip if available, otherwise palm center
            if "index_finger_tip" in self.hand:
                current_head = (
                    int(self.hand["index_finger_tip"]["x"] * GAME_WIDTH / 640),
                    int(self.hand["index_finger_tip"]["y"] * GAME_HEIGHT / 480)
                )
            else:
                current_head = (
                    int(self.hand["position"]["x"] * GAME_WIDTH / 640),
                    int(self.hand["position"]["y"] * GAME_HEIGHT / 480)
                )
            
            # Update snake movement
            if self.previous_head:
                px, py = self.previous_head
                cx, cy = current_head
                
                self.points.append([cx, cy])
                distance = math.hypot(cx - px, cy - py)
                
                # Apply speed based on difficulty
                speed_multiplier = DIFFICULTY_LEVELS[self.difficulty]["speed_multiplier"]
                adjusted_distance = distance * speed_multiplier
                
                self.lengths.append(adjusted_distance)
                self.current_length += adjusted_distance
                
                # Length reduction
                if self.current_length > self.allowed_length:
                    for i, length in enumerate(self.lengths):
                        self.current_length -= length
                        self.lengths.pop(i)
                        self.points.pop(i)
                        if self.current_length < self.allowed_length:
                            break
                
                # Check if snake ate food
                rx, ry = self.food_point
                if (rx - FOOD_SIZE // 2 < cx < rx + FOOD_SIZE // 2 and
                    ry - FOOD_SIZE // 2 < cy < ry + FOOD_SIZE // 2):
                    self.random_food_location()  # This will also select a new random food image
                    self.allowed_length += SNAKE_GROWTH
                    score_multiplier = DIFFICULTY_LEVELS[self.difficulty]["score_multiplier"]
                    self.score += int(10 * score_multiplier)
                    print(f"Food eaten! Score: {self.score}")
                
                # Check for self collision
                if len(self.points) > 10:  # Only check when snake is long enough
                    pts = np.array(self.points[:-5], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    if len(pts) > 0:
                        min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)
                        if -1 <= min_dist <= 1:
                            print("Snake collision! Game over!")
                            self.game_over = True
                            self.game_active = False
                            self.save_game_result()
                            return
            
            self.previous_head = current_head
    
    def calculate_skill_metrics(self):
        """Calculate skill metrics based on game performance"""
        metrics = {}
        
        # Hand-eye coordination: Based on score and snake length control
        if self.score > 0:
            metrics["hand_eye_coordination"] = min(100, (self.score / 50) * 100)
        else:
            metrics["hand_eye_coordination"] = 0
        
        # Agility: Based on snake length and movement control
        if self.allowed_length > INITIAL_SNAKE_LENGTH:
            agility_factor = ((self.allowed_length - INITIAL_SNAKE_LENGTH) / 500) * 100
            metrics["agility"] = min(100, agility_factor)
        else:
            metrics["agility"] = 0
        
        # Focus: Based on game duration without collision
        if self.game_over and self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            focus_factor = (duration / GAME_DURATION) * 100
            metrics["focus"] = min(100, focus_factor)
        else:
            metrics["focus"] = 0
        
        # Reaction time: Based on food collection speed
        if self.score > 0:
            reaction_factor = (self.score / 20) * 100
            metrics["reaction_time"] = min(100, reaction_factor)
        else:
            metrics["reaction_time"] = 0
        
        return metrics
    
    def save_game_result(self):
        """Save the game result for reporting"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate skill metrics
            skill_metrics = self.calculate_skill_metrics()
            
            # Create game result object
            result = {
                "game_id": self.game_id,
                "game_name": "Snake",
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
            
            # Keep only the last 100 results
            if len(game_results) > 100:
                game_results.pop(0)
            
            # Save to database using Prisma
            import asyncio
            asyncio.create_task(self._persist_to_database(result, skill_metrics))
    
    async def _persist_to_database(self, result, skill_metrics):
        """Persist game result to database using Prisma"""
        try:
            from database import prisma
            
            if not prisma.is_connected():
                await prisma.connect()
            
            game_type_id = "snake"
            
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
            
            print(f"Saved snake game report to database with ID: {game_report.id}")
            
        except Exception as e:
            print(f"Error saving snake game report to database: {str(e)}")
    
    def render_frame(self):
        """Render the current game state to an image"""
        # Start with camera frame as background if available
        if self.current_camera_frame is not None and len(self.current_camera_frame) > 0:
            try:
                if isinstance(self.current_camera_frame, str):
                    image_data = base64.b64decode(self.current_camera_frame.split(',')[1] if ',' in self.current_camera_frame else self.current_camera_frame)
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    img = self.current_camera_frame
                
                if img is not None:
                    img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))
                else:
                    img = self._create_background()
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                img = self._create_background()
        else:
            img = self._create_background()
        
        # Apply slight overlay
        overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        overlay.fill(30)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(img, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 255, 0), SNAKE_THICKNESS)
            
            # Draw head
            if len(self.points) > 0:
                cv2.circle(img, tuple(self.points[-1]), SNAKE_THICKNESS // 2, (0, 255, 0), cv2.FILLED)
        
        # Draw food
        if self.food_point:
            fx, fy = self.food_point
            
            if self.current_food_image is not None and len(self.current_food_image.shape) > 0:
                # Use the randomly selected food image
                try:
                    # Check if image has alpha channel
                    if self.current_food_image.shape[2] == 4:
                        # Separate alpha channel
                        bgr = self.current_food_image[:, :, :3]
                        alpha = self.current_food_image[:, :, 3]
                        
                        # Calculate position for overlay
                        x1 = max(0, fx - FOOD_SIZE // 2)
                        y1 = max(0, fy - FOOD_SIZE // 2)
                        x2 = min(GAME_WIDTH, x1 + FOOD_SIZE)
                        y2 = min(GAME_HEIGHT, y1 + FOOD_SIZE)
                        
                        # Adjust image size if it goes out of bounds
                        img_x1 = 0 if x1 >= 0 else -x1
                        img_y1 = 0 if y1 >= 0 else -y1
                        img_x2 = FOOD_SIZE - (x2 - min(GAME_WIDTH, x1 + FOOD_SIZE))
                        img_y2 = FOOD_SIZE - (y2 - min(GAME_HEIGHT, y1 + FOOD_SIZE))
                        
                        # Crop the image and alpha if needed
                        bgr_crop = bgr[img_y1:img_y2, img_x1:img_x2]
                        alpha_crop = alpha[img_y1:img_y2, img_x1:img_x2]
                        
                        # Create mask
                        mask = alpha_crop.astype(float) / 255.0
                        mask_3channel = np.dstack((mask, mask, mask))
                        
                        # Get the region of interest from the main image
                        roi = img[y1:y2, x1:x2]
                        
                        # Blend the images
                        blended = (bgr_crop * mask_3channel + roi * (1 - mask_3channel)).astype(np.uint8)
                        
                        # Put the blended image back
                        img[y1:y2, x1:x2] = blended
                    else:
                        # No alpha channel, just place the image
                        x1 = max(0, fx - FOOD_SIZE // 2)
                        y1 = max(0, fy - FOOD_SIZE // 2)
                        x2 = min(GAME_WIDTH, x1 + FOOD_SIZE)
                        y2 = min(GAME_HEIGHT, y1 + FOOD_SIZE)
                        
                        img[y1:y2, x1:x2] = self.current_food_image[0:y2-y1, 0:x2-x1]
                except Exception as e:
                    print(f"Error drawing food image: {e}")
                    # Fallback to drawing circles if image fails
                    cv2.circle(img, (fx, fy), FOOD_SIZE // 2, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (fx, fy), FOOD_SIZE // 2 - 5, (255, 0, 0), cv2.FILLED)
            else:
                # Fallback to drawing circles if image not loaded
                cv2.circle(img, (fx, fy), FOOD_SIZE // 2, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (fx, fy), FOOD_SIZE // 2 - 5, (255, 0, 0), cv2.FILLED)
        
        
        # Draw HUD
        # Score
        cv2.putText(img, f"Score: {self.score}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Time
        cv2.putText(img, f"Time: {self.time_remaining}s", (GAME_WIDTH - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Difficulty
        cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH // 2 - 80, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Game over screen
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            cv2.putText(img, "GAME OVER", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH//2 - 120, GAME_HEIGHT//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH//2 - 100, GAME_HEIGHT//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to base64
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        return None
    
    def _create_background(self):
        """Create a background for the game"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        
        # Create a gradient background
        for y in range(GAME_HEIGHT):
            green_value = int(150 - (y / GAME_HEIGHT) * 50)
            cv2.line(img, (0, y), (GAME_WIDTH, y), (0, green_value, 0), 1)
        
        return img