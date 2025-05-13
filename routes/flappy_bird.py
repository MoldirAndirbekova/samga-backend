# games/flappy_bird.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
import cv2
import random
import base64
import math
import time

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
BIRD_WIDTH = 68
BIRD_HEIGHT = 48
PIPE_WIDTH = 104
PIPE_HEIGHT = 640
PIPE_GAP = 200
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 3
GROUND_HEIGHT = 112

DIFFICULTY_LEVELS = {
    "EASY": {
        "pipe_gap": 250,
        "pipe_speed": 2,
        "spawn_rate": 3.0  # seconds
    },
    "MEDIUM": {
        "pipe_gap": 200,
        "pipe_speed": 3,
        "spawn_rate": 2.5
    },
    "HARD": {
        "pipe_gap": 150,
        "pipe_speed": 4,
        "spawn_rate": 2.0
    }
}

class Bird:
    def __init__(self):
        self.x = GAME_WIDTH // 4
        self.y = GAME_HEIGHT // 2
        self.velocity = 0
        self.animation_index = 0
        self.animation_counter = 0
        self.angle = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Update angle based on velocity
        if self.velocity < 0:
            self.angle = min(self.angle + 3, 30)  # Tilting up
        else:
            self.angle = max(self.angle - 3, -90)  # Tilting down
        
        # Animate bird
        self.animation_counter += 1
        if self.animation_counter % 5 == 0:
            self.animation_index = (self.animation_index + 1) % 3

class Pipe:
    def __init__(self, x, gap_y, gap_size):
        self.x = x
        self.gap_y = gap_y
        self.gap_size = gap_size
        self.passed = False

    def get_rects(self):
        # Top pipe
        top_rect = (self.x, 0, PIPE_WIDTH, self.gap_y)
        # Bottom pipe
        bottom_rect = (self.x, self.gap_y + self.gap_size, PIPE_WIDTH, GAME_HEIGHT - (self.gap_y + self.gap_size))
        return top_rect, bottom_rect

class FlappyBirdGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id
        
        # Game state
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        self.last_pipe_spawn = None
        
        # Difficulty settings
        self.pipe_gap = DIFFICULTY_LEVELS[difficulty]["pipe_gap"]
        self.pipe_speed = DIFFICULTY_LEVELS[difficulty]["pipe_speed"]
        self.spawn_rate = DIFFICULTY_LEVELS[difficulty]["spawn_rate"]
        
        # Pose tracking
        self.pose_detected = False
        self.arm_raised = False
        self.arm_angle_left = 0
        self.arm_angle_right = 0
        
        # Camera frame storage for AR overlay
        self.current_camera_frame = None
        
        # Assets (will be loaded in a real implementation)
        self.background = None
        self.bird_images = []
        self.pipe_image = None
        self.ground_image = None
        
        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps
        
        # Load assets
        self.load_assets()

    def load_assets(self):
        """Load game assets from static/images directory"""
        try:
            # Attempt to load images
            assets_path = "/app/static/images"
            
            # Load bird animations
            self.bird_images = []
            for state in ['downflap', 'midflap', 'upflap']:
                bird_path = f"{assets_path}/yellowbird-{state}.png"
                bird_img = cv2.imread(bird_path, cv2.IMREAD_UNCHANGED)
                if bird_img is not None:
                    # Scale bird image if needed
                    bird_img = cv2.resize(bird_img, (BIRD_WIDTH, BIRD_HEIGHT))
                    self.bird_images.append(bird_img)
            
            # Load pipe
            pipe_path = f"{assets_path}/pipe-green.png"
            self.pipe_image = cv2.imread(pipe_path, cv2.IMREAD_UNCHANGED)
            if self.pipe_image is not None:
                self.pipe_image = cv2.resize(self.pipe_image, (PIPE_WIDTH, PIPE_HEIGHT))
            
            # Load background
            bg_path = f"{assets_path}/background-night.png"
            self.background = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
            if self.background is not None:
                self.background = cv2.resize(self.background, (GAME_WIDTH, GAME_HEIGHT))
            
            # Load ground
            ground_path = f"{assets_path}/floor.png"
            self.ground_image = cv2.imread(ground_path, cv2.IMREAD_UNCHANGED)
            if self.ground_image is not None:
                self.ground_image = cv2.resize(self.ground_image, (GAME_WIDTH, GROUND_HEIGHT))
            
            print("Assets loaded successfully")
        except Exception as e:
            print(f"Error loading assets: {e}")
    
    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Flappy Bird game id: {self.game_id} with difficulty: {self.difficulty}")
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.last_pipe_spawn = datetime.now()
        self.score = 0
        self.bird = Bird()
        self.pipes = []
    
    # In flappy_bird.py, modify the update_pose method:
    def update_pose(self, pose_data):
        """Update game based on pose detection"""
        if not pose_data or not pose_data.get("pose_detected", False):
            self.pose_detected = False
            return
        
        self.pose_detected = True
        
        # Check if both arms are raised
        left_arm_raised = pose_data.get("left_arm_raised", False)
        right_arm_raised = pose_data.get("right_arm_raised", False)
        
        # Store arm angles for debugging or visualization
        self.arm_angle_left = pose_data.get("left_arm_angle", 0)
        self.arm_angle_right = pose_data.get("right_arm_angle", 0)
        
        # Detect flap action - both arms raised
        arms_raised = left_arm_raised and right_arm_raised
        
        # Flap the bird if arms were down but now raised (transition)
        if arms_raised and not self.arm_raised and self.game_active:
            self.bird.flap()
        
        # Update arm raised state
        self.arm_raised = arms_raised
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        angle = math.degrees(
            math.atan2(p3['y'] - p2['y'], p3['x'] - p2['x']) -
            math.atan2(p1['y'] - p2['y'], p1['x'] - p2['x'])
        )
        return abs(angle)
    
    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame
    
    def spawn_pipe(self):
        """Spawn a new pipe"""
        gap_y = random.randint(100, GAME_HEIGHT - self.pipe_gap - 100)
        new_pipe = Pipe(GAME_WIDTH, gap_y, self.pipe_gap)
        self.pipes.append(new_pipe)
    
    def check_collision(self):
        """Check if bird collides with pipes or boundaries"""
        bird_rect = (self.bird.x - BIRD_WIDTH//2, self.bird.y - BIRD_HEIGHT//2, 
                     BIRD_WIDTH, BIRD_HEIGHT)
        
        # Check ground collision
        if self.bird.y + BIRD_HEIGHT//2 >= GAME_HEIGHT - GROUND_HEIGHT:
            return True
        
        # Check ceiling collision
        if self.bird.y - BIRD_HEIGHT//2 <= 0:
            return True
        
        # Check pipe collision
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if self.rect_collision(bird_rect, top_rect) or self.rect_collision(bird_rect, bottom_rect):
                return True
        
        return False
    
    def rect_collision(self, rect1, rect2):
        """Check if two rectangles collide"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return (x1 < x2 + w2 and x1 + w1 > x2 and 
                y1 < y2 + h2 and y1 + h1 > y2)
    
    def update_game_state(self):
        """Update the game state for one frame"""
        if not self.game_active or self.game_over:
            return
        
        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now
        
        # Update bird
        self.bird.update()
        
        # Check collision
        if self.check_collision():
            self.game_over = True
            self.game_active = False
            self.save_game_result()
            return
        
        # Spawn pipes
        if (now - self.last_pipe_spawn).total_seconds() > self.spawn_rate:
            self.spawn_pipe()
            self.last_pipe_spawn = now
        
        # Update pipes
        for pipe in self.pipes:
            pipe.x -= self.pipe_speed
            
            # Check if bird passed pipe
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
        
        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe.x > -PIPE_WIDTH]
    
    def calculate_skill_metrics(self):
        """Calculate skill metrics based on game performance"""
        metrics = {}
        
        # Hand-eye coordination
        if self.score > 0:
            metrics["hand_eye_coordination"] = min(100, (self.score / 20) * 100)
        else:
            metrics["hand_eye_coordination"] = 0
        
        # Agility (based on successful flaps)
        if self.score > 0:
            metrics["agility"] = min(100, (self.score / 15) * 100)
        else:
            metrics["agility"] = 0
        
        # Focus (based on duration)
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            metrics["focus"] = min(100, (duration / 120) * 100)
        else:
            metrics["focus"] = 0
        
        # Reaction time (based on score per minute)
        if self.start_time and self.score > 0:
            duration = (datetime.now() - self.start_time).total_seconds()
            score_per_minute = (self.score / duration) * 60
            metrics["reaction_time"] = min(100, score_per_minute * 5)
        else:
            metrics["reaction_time"] = 0
        
        return metrics
    
    def save_game_result(self):
        """Save the game result for reporting"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            skill_metrics = self.calculate_skill_metrics()
            
            result = {
                "game_id": self.game_id,
                "game_name": "Flappy Bird",
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
            
            if len(game_results) > 100:
                game_results.pop(0)
            
            import asyncio
            asyncio.create_task(self._persist_to_database(result, skill_metrics))
    
    async def _persist_to_database(self, result, skill_metrics):
        """Persist game result to database using Prisma"""
        try:
            from database import prisma
            
            if not prisma.is_connected():
                await prisma.connect()
            
            game_type_id = "flappy-bird"
            
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
            
            print(f"Saved flappy bird game report to database with ID: {game_report.id}")
            
        except Exception as e:
            print(f"Error saving flappy bird game report to database: {str(e)}")
    
    def render_frame(self):
        """Render the current game state to an image"""
        # Start with camera frame or background
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
        
        # Draw background image if available
        if self.background is not None:
            img = self.overlay_image(img, self.background, 0, 0)
        
        # Draw pipes
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            
            if self.pipe_image is not None:
                # Draw top pipe (flipped)
                flipped_pipe = cv2.flip(self.pipe_image, 0)
                img = self.overlay_image(img, flipped_pipe, top_rect[0], top_rect[1] + top_rect[3] - PIPE_HEIGHT)
                
                # Draw bottom pipe
                img = self.overlay_image(img, self.pipe_image, bottom_rect[0], bottom_rect[1])
            else:
                # Fallback to rectangles
                cv2.rectangle(img, (top_rect[0], top_rect[1]), 
                            (top_rect[0] + top_rect[2], top_rect[1] + top_rect[3]), 
                            (0, 128, 0), -1)
                cv2.rectangle(img, (bottom_rect[0], bottom_rect[1]), 
                            (bottom_rect[0] + bottom_rect[2], bottom_rect[1] + bottom_rect[3]), 
                            (0, 128, 0), -1)
        
        # Draw ground
        if self.ground_image is not None:
            img = self.overlay_image(img, self.ground_image, 0, GAME_HEIGHT - GROUND_HEIGHT)
        else:
            cv2.rectangle(img, (0, GAME_HEIGHT - GROUND_HEIGHT), 
                        (GAME_WIDTH, GAME_HEIGHT), (139, 69, 19), -1)
        
        # Draw bird
        if self.bird_images and self.bird.animation_index < len(self.bird_images):
            bird_img = self.bird_images[self.bird.animation_index]
            
            # Rotate bird based on angle
            rotated_bird = self.rotate_image(bird_img, self.bird.angle)
            
            # Draw bird
            bird_x = int(self.bird.x - BIRD_WIDTH // 2)
            bird_y = int(self.bird.y - BIRD_HEIGHT // 2)
            img = self.overlay_image(img, rotated_bird, bird_x, bird_y)
        else:
            # Fallback to circle
            cv2.circle(img, (int(self.bird.x), int(self.bird.y)), 15, (255, 255, 0), -1)
        
        # Draw HUD
        # Score
        cv2.putText(img, f"Score: {self.score}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Difficulty
        cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH - 200, 40), 
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
    
    def overlay_image(self, background, overlay, x, y):
        """Overlay image with transparency"""
        h, w = overlay.shape[:2]
        
        # Check bounds
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return background
        
        if len(overlay.shape) == 3 and overlay.shape[2] == 4:
            # Image has alpha channel
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + 
                                              (1 - alpha) * background[y:y+h, x:x+w, c])
        else:
            # No alpha channel
            background[y:y+h, x:x+w] = overlay
        
        return background
    
    def rotate_image(self, image, angle):
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_TRANSPARENT)
        return rotated
    
    def _create_background(self):
        """Create a default background"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        img[:] = (135, 206, 235)  # Sky blue
        return img