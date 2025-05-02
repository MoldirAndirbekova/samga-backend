# games/fruit_slicer.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
import cv2
import random
import base64
import math

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
FRUIT_SIZE_MIN = 80
FRUIT_SIZE_MAX = 150
GAME_DURATION = 60  # 60 seconds

# Difficulty levels
DIFFICULTY_LEVELS = {
    "EASY": {
        "fruit_lifetime": 6.0,  # seconds
        "score_penalty": 1,
        "spawn_rate": (4.0, 6.0),  # min and max time between spawns
        "bomb_chance": 0.1  # 10% chance of bombs
    },
    "MEDIUM": {
        "fruit_lifetime": 4.0,
        "score_penalty": 2,
        "spawn_rate": (2.0, 3.5),
        "bomb_chance": 0.3
    },
    "HARD": {
        "fruit_lifetime": 2.5,
        "score_penalty": 3,
        "spawn_rate": (1.5, 3.0),
        "bomb_chance": 0.5
    }
}

class Fruit:
    def __init__(self, id, x, y, size, created_at, lifetime, is_bomb=False):
        self.id = id
        self.x = x
        self.y = y
        self.size = size
        self.sliced = False
        self.sliced_by_player = False
        self.age = 0
        self.created_at = created_at
        self.lifetime = lifetime
        self.is_bomb = is_bomb
        self.velocity_x = random.uniform(-3, 3)
        self.velocity_y = random.uniform(8, 12)  # Initial upward velocity
        self.gravity = 0.3
        self.rotation = 0
        self.rotation_speed = random.uniform(-5, 5)
        
    def update_position(self, dt):
        # Apply gravity (negative to pull DOWN)
        self.velocity_y -= self.gravity * dt * 60
        
        # Update position 
        self.x += self.velocity_x * dt * 60
        self.y += self.velocity_y * dt * 60  # Positive velocity = upward
        
        # Update rotation
        self.rotation += self.rotation_speed * dt * 60
        
        # Keep rotation between 0-360
        self.rotation %= 360

class FruitSlicerGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id
        if difficulty not in DIFFICULTY_LEVELS:
            self.difficulty = "MEDIUM"  # Default to medium if invalid
        
        # Game state
        self.score = 0
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        self.time_remaining = GAME_DURATION
        self.fruits = []
        self.fruit_id_counter = 0
        self.last_fruit_spawn = datetime.now()
        self.penalties = 0
        self.combo = 0
        self.max_combo = 0
        
        self.nose_position = None
        self.previous_nose_position = None
        
        # Camera frame storage for AR overlay
        self.current_camera_frame = None
        
        # Blade trail for visual effect
        self.blade_trail = []
        
        # Game physics
        self.fps = 60
        self.frame_time = 1 / self.fps

    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Fruit Slicer game id: {self.game_id} with difficulty: {self.difficulty}")
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.last_fruit_spawn = datetime.now()
        self.score = 0
        self.penalties = 0
        self.combo = 0
        self.max_combo = 0
        self.time_remaining = GAME_DURATION
        self.fruits = []
        self.fruit_id_counter = 0
        self.nose_position = None
        self.previous_nose_position = None
        self.blade_trail = []
    
    def update_pose(self, pose_data):
        """Update nose position from pose tracking data"""
        self.previous_nose_position = self.nose_position
        self.nose_position = pose_data
        
        # Add to blade trail if nose is moving
        if pose_data:
            self.blade_trail.append({
                "x": pose_data["x"],
                "y": pose_data["y"],
                "time": datetime.now()
            })
            
        # Only keep recent points for the trail (last 0.5 seconds)
        now = datetime.now()
        self.blade_trail = [p for p in self.blade_trail 
                            if (now - p["time"]).total_seconds() < 0.5]
    
    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame
    
    def spawn_fruit(self):
        """Spawn a new fruit or bomb"""
        # Determine if this should be a bomb based on difficulty
        is_bomb = random.random() < DIFFICULTY_LEVELS[self.difficulty]["bomb_chance"]
        
        lifetime = DIFFICULTY_LEVELS[self.difficulty]["fruit_lifetime"]
        
        # Start fruits from bottom and throw them upward
        new_fruit = Fruit(
            id=self.fruit_id_counter,
            x=random.uniform(100, GAME_WIDTH - 100),
            y=GAME_HEIGHT,  # Start at BOTTOM of screen
            size=random.uniform(FRUIT_SIZE_MIN, FRUIT_SIZE_MAX),
            created_at=datetime.now(),
            lifetime=lifetime,
            is_bomb=is_bomb
        )
        
        # Set velocity to go upward with reduced gravity for slower falls
        new_fruit.velocity_y = random.uniform(12, 18)  # Higher initial velocity
        new_fruit.velocity_x = random.uniform(-2, 2)   # Less horizontal movement
        new_fruit.gravity = 0.15  # Reduced gravity (was 0.3)
        
        self.fruits.append(new_fruit)
        self.fruit_id_counter += 1
        print(f"Spawned {'bomb' if is_bomb else 'fruit'} {self.fruit_id_counter} at ({new_fruit.x}, {new_fruit.y}) with velocity ({new_fruit.velocity_x}, {new_fruit.velocity_y})")
    
    def check_fruit_slices(self):
        """Check if nose has directly touched any fruits"""
        if not self.nose_position:
            return
        
        sliced_fruits = []
        
        # Get nose position in game coordinates
        scale_x = GAME_WIDTH / 640
        scale_y = GAME_HEIGHT / 480
        
        nose_x = self.nose_position["x"] * scale_x
        nose_y = self.nose_position["y"] * scale_y
        
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            
            # Get fruit center and calculate distance to nose
            fruit_center_x = fruit.x + fruit.size / 2
            fruit_center_y = fruit.y + fruit.size / 2
            
            # Calculate direct distance from nose to fruit center
            distance = math.sqrt((nose_x - fruit_center_x)**2 + (nose_y - fruit_center_y)**2)
            
            # Check if nose is inside the fruit (direct contact)
            if distance < fruit.size / 2:  # Only slice if nose is inside fruit radius
                fruit.sliced = True
                fruit.sliced_by_player = True
                sliced_fruits.append(fruit)
                print(f"Sliced fruit with nose at ({nose_x:.1f}, {nose_y:.1f}), fruit at ({fruit_center_x:.1f}, {fruit_center_y:.1f})")
        
        # Rest of the score update logic remains the same
        if sliced_fruits:
            bomb_hit = False
            points_earned = 0
            
            for fruit in sliced_fruits:
                if fruit.is_bomb:
                    bomb_hit = True
                    break
                else:
                    points_earned += 1
            
            if bomb_hit:
                self.combo = 0
                self.penalties += 5
                print("Bomb hit! Penalty applied.")
                
                # Game over on bomb hit
                self.game_over = True
                self.game_active = False
                self.save_game_result()
            else:
                self.score += points_earned
                self.combo += points_earned
                self.max_combo = max(self.max_combo, self.combo)
                print(f"Score increased to {self.score}, combo: {self.combo}")
    
    def check_fruit_boundaries(self):
        """Check if any fruits have fallen off screen or exceeded lifetime"""
        now = datetime.now()
        missed_fruits = []
        
        for fruit in self.fruits:
            if fruit.sliced:
                continue
                
            # Check if fruit has fallen off screen
            if fruit.y < 0 and fruit.velocity_y < 0:
                fruit.sliced = True
                if not fruit.is_bomb:
                    missed_fruits.append(fruit)
        
        # Apply penalties for missed fruits
        if missed_fruits:
            self.combo = 0
            penalty = len(missed_fruits) * DIFFICULTY_LEVELS[self.difficulty]["score_penalty"]
            self.penalties += penalty
            self.score = max(0, self.score - penalty)
            print(f"{len(missed_fruits)} fruits missed! Penalty: -{penalty}, New score: {self.score}")
    
    def update_fruit_positions(self, dt):
        """Update positions of all fruits based on physics"""
        for fruit in self.fruits:
            if not fruit.sliced:
                fruit.update_position(dt)
            else:
                # If sliced, continue animation but split in two
                fruit.age += 1
        
        # Clean up old fruits
        initial_count = len(self.fruits)
        new_fruits = []
        for fruit in self.fruits:
            # Keep unsliced fruits, or recently sliced fruits for visual feedback
            if not fruit.sliced or fruit.age < 20:
                new_fruits.append(fruit)
                if fruit.sliced:
                    fruit.age += 1
        
        self.fruits = new_fruits
        if initial_count != len(self.fruits):
            print(f"Cleaned fruits: {initial_count} -> {len(self.fruits)}")
    
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
        
        # Make sure we're spawning fruits - add debug logging
        print(f"Time since last fruit spawn: {(now - self.last_fruit_spawn).total_seconds()}")
        
        # Force spawn a fruit if none exist and game just started
        if len(self.fruits) == 0 and self.time_remaining > 58:
            print("Forcing initial fruit spawn")
            self.spawn_fruit()
            self.last_fruit_spawn = now
        
        # Spawn new fruits at rate based on difficulty
        min_spawn_time, max_spawn_time = DIFFICULTY_LEVELS[self.difficulty]["spawn_rate"]
        fruit_spawn_time = (now - self.last_fruit_spawn).total_seconds()
        if fruit_spawn_time > random.uniform(min_spawn_time, max_spawn_time):
            print(f"Spawning new fruit after {fruit_spawn_time} seconds")
            self.spawn_fruit()
            self.last_fruit_spawn = now
        else:
            print(f"Not spawning yet, waiting {fruit_spawn_time}/{min_spawn_time}-{max_spawn_time}")
        
        # Update fruit positions
        self.update_fruit_positions(dt)
        
        # Check for fruit slices
        self.check_fruit_slices()
        
        # Check for missed fruits
        self.check_fruit_boundaries()
    
    def calculate_skill_metrics(self):
        """Calculate skill metrics based on game performance"""
        metrics = {}
        
        # Hand-eye coordination: Based on successful slices vs penalties
        total_attempts = self.score + self.penalties
        if total_attempts > 0:
            metrics["hand_eye_coordination"] = min(100, (self.score / total_attempts) * 100)
        else:
            metrics["hand_eye_coordination"] = 0
            
        # Agility: Based on score within time limit
        agility_factor = (self.score / 50) * 100  # 50 slices in 60 sec = 100% agility
        metrics["agility"] = min(100, agility_factor)
            
        # Focus: Inverse of penalties (fewer penalties = better focus)
        if self.penalties == 0 and self.score > 0:
            metrics["focus"] = 100
        elif total_attempts > 0:
            focus_factor = 100 - ((self.penalties / total_attempts) * 100)
            metrics["focus"] = max(0, focus_factor)
        else:
            metrics["focus"] = 0
            
        # Reaction time: Based on max combo (higher combo = better reaction)
        reaction_factor = (self.max_combo / 10) * 100  # 10 combo = 100% reaction
        metrics["reaction_time"] = min(100, reaction_factor)
        
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
                "difficulty": self.difficulty,
                "score": self.score,
                "duration_seconds": int(duration),
                "left_score": self.score,  # Use score as left_score for compatibility
                "right_score": self.penalties,  # Use penalties as right_score for reporting
                "timestamp": datetime.now().isoformat(),
                "skills": skill_metrics,
                "child_id": self.child_id
            }
            
            # Add to global results list (from games.py)
            from .games import game_results
            game_results.append(result)
            print(f"Saved game result: difficulty={self.difficulty}, score={self.score}, penalties={self.penalties}, duration={duration}")
            
            # Keep only the last 100 results to avoid memory issues
            if len(game_results) > 100:
                game_results.pop(0)
    
    def render_frame(self):
        """Render the current game state to an image"""
        print(f"Rendering frame: active={self.game_active}, score={self.score}, fruits={len(self.fruits)}")
        
        # Start with camera frame as background if available
        if self.current_camera_frame is not None and len(self.current_camera_frame) > 0:
            try:
                # Decode and process camera frame
                if isinstance(self.current_camera_frame, str):
                    # If we received base64 string
                    image_data = base64.b64decode(self.current_camera_frame.split(',')[1] if ',' in self.current_camera_frame else self.current_camera_frame)
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    img = self.current_camera_frame
                
                # Resize to game dimensions
                if img is not None:
                    img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))
                else:
                    # Fallback to gradient background
                    img = self._create_gradient_background()
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                img = self._create_gradient_background()
        else:
            # Fallback to gradient background
            img = self._create_gradient_background()
        
        # Apply slight overlay color for game atmosphere
        overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        overlay.fill(20)  # Dark overlay
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Draw blade trail
        if len(self.blade_trail) > 1:
            scale_x = GAME_WIDTH / 640
            scale_y = GAME_HEIGHT / 480
            
            for i in range(1, len(self.blade_trail)):
                pt1 = (
                    int(self.blade_trail[i-1]["x"] * scale_x),
                    int(self.blade_trail[i-1]["y"] * scale_y)
                )
                pt2 = (
                    int(self.blade_trail[i]["x"] * scale_x),
                    int(self.blade_trail[i]["y"] * scale_y)
                )
                
                # Calculate alpha based on time
                time_diff = (datetime.now() - self.blade_trail[i]["time"]).total_seconds()
                alpha = max(0, 1 - time_diff / 0.5)
                
                # Draw trail with glow effect
                thickness = int(5 * alpha)
                if thickness > 0:
                    # Yellow-orange trail for nose
                    cv2.line(img, pt1, pt2, (0, 200, 255), thickness + 4)
                    # Core
                    cv2.line(img, pt1, pt2, (0, 255, 255), thickness)
        
        # Draw nose position if detected
        if self.nose_position:
            scale_x = GAME_WIDTH / 640
            scale_y = GAME_HEIGHT / 480
            
            nose_x = int(self.nose_position["x"] * scale_x)
            nose_y = int(self.nose_position["y"] * scale_y)
            
            # Draw a precise cursor for the nose
            cursor_radius = 15  # Size of the cursor (same size used for hit detection)
            
            # Draw outer circle
            cv2.circle(img, (nose_x, nose_y), cursor_radius, (0, 0, 0), 2)  # Black outline
            # Draw inner filled circle
            cv2.circle(img, (nose_x, nose_y), cursor_radius - 2, (0, 255, 255), -1)  # Yellow fill
            
            # Add crosshairs for precision
            cv2.line(img, (nose_x - cursor_radius, nose_y), (nose_x + cursor_radius, nose_y), (0, 0, 0), 1)
            cv2.line(img, (nose_x, nose_y - cursor_radius), (nose_x, nose_y + cursor_radius), (0, 0, 0), 1)
            
        # Draw fruits
        for fruit in self.fruits:
            center = (int(fruit.x + fruit.size/2), int(fruit.y + fruit.size/2))
            radius = int(fruit.size/2)
            
            # Create rotation matrix for fruit
            M = cv2.getRotationMatrix2D(center, fruit.rotation, 1)
            
            if fruit.sliced:
                if fruit.sliced_by_player:
                    # Fruit was sliced - show split effect
                    if fruit.is_bomb:
                        # Bomb explosion - red
                        alpha = max(0, 1 - fruit.age / 20)
                        
                        # Draw explosion rings
                        for r in range(radius, radius*3, radius//2):
                            cv2.circle(img, center, r, (50, 50, 255, int(alpha * 255)), 5)
                        
                        # Add text
                        cv2.putText(img, "BOOM!", 
                                  (center[0]-40, center[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  (50, 50, 255, int(alpha * 255)), 2)
                    else:
                        # Normal fruit - show sliced halves
                        alpha = max(0, 1 - fruit.age / 20)
                        
                        # Calculate direction perpendicular to slice
                        slice_dir_x = 0
                        slice_dir_y = -1  # Default upward
                        
                        # Draw two halves
                        half1_center = (
                            int(center[0] + slice_dir_x * radius * 0.7 * fruit.age/10),
                            int(center[1] + slice_dir_y * radius * 0.7 * fruit.age/10)
                        )
                        half2_center = (
                            int(center[0] - slice_dir_x * radius * 0.7 * fruit.age/10),
                            int(center[1] - slice_dir_y * radius * 0.7 * fruit.age/10)
                        )
                        
                        # Draw with alpha fade
                        fruit_color = (50, 255, 50) if not fruit.is_bomb else (50, 50, 255)
                        
                        # Draw ellipses for halves
                        cv2.ellipse(img, half1_center, (radius, radius//2), 
                                  fruit.rotation, 0, 180, fruit_color, -1)
                        cv2.ellipse(img, half2_center, (radius, radius//2), 
                                  fruit.rotation + 180, 0, 180, fruit_color, -1)
                        
                        # Add juice drips
                        for i in range(3):
                            drip_x = center[0] + random.randint(-radius, radius)
                            drip_y = center[1] + random.randint(0, fruit.age*2)
                            drip_length = random.randint(5, 10)
                            cv2.line(img, (drip_x, drip_y), 
                                   (drip_x, drip_y + drip_length), 
                                   fruit_color, 2)
            else:
                # Draw intact fruit/bomb with gradient
                if fruit.is_bomb:
                    # Bomb
                    cv2.circle(img, center, radius, (50, 50, 200), -1)
                    cv2.circle(img, center, radius, (0, 0, 0), 2)
                    
                    # Fuse
                    fuse_top = (center[0], center[1] - radius - 10)
                    cv2.line(img, (center[0], center[1] - radius), 
                           fuse_top, (100, 100, 100), 3)
                    
                    # Fuse spark
                    if random.random() > 0.5:
                        spark_radius = random.randint(2, 5)
                        spark_color = random.choice([
                            (0, 0, 255), (0, 255, 255), (255, 255, 0)
                        ])
                        cv2.circle(img, fuse_top, spark_radius, spark_color, -1)
                    
                    # Bomb pattern
                    cv2.putText(img, "BOMB", 
                              (center[0]-30, center[1]+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 1)
                else:
                    # Normal fruit
                    fruit_type = fruit.id % 5  # 5 different fruit types
                    
                    if fruit_type == 0:  # Apple
                        fruit_color = (0, 0, 255)  # Red in BGR
                        cv2.circle(img, center, radius, fruit_color, -1)
                        
                        # Stem
                        stem_top = (center[0], center[1] - radius - 5)
                        cv2.line(img, (center[0], center[1] - radius), 
                               stem_top, (40, 100, 40), 2)
                    
                    elif fruit_type == 1:  # Orange
                        fruit_color = (0, 165, 255)  # Orange in BGR
                        cv2.circle(img, center, radius, fruit_color, -1)
                        
                        # Texture circles
                        for i in range(5):
                            x = center[0] + int(radius * 0.6 * math.cos(i * math.pi/2.5))
                            y = center[1] + int(radius * 0.6 * math.sin(i * math.pi/2.5))
                            cv2.circle(img, (x, y), 3, (0, 140, 220), 1)
                    
                    elif fruit_type == 2:  # Watermelon
                        fruit_color = (30, 180, 30)  # Green in BGR
                        cv2.circle(img, center, radius, fruit_color, -1)
                        
                        # Stripes
                        for i in range(0, 360, 30):
                            rad = i * math.pi / 180
                            pt1 = (
                                int(center[0] - radius * math.cos(rad)),
                                int(center[1] - radius * math.sin(rad))
                            )
                            pt2 = (
                                int(center[0] + radius * math.cos(rad)),
                                int(center[1] + radius * math.sin(rad))
                            )
                            cv2.line(img, pt1, pt2, (30, 120, 30), 1)
                    
                    elif fruit_type == 3:  # Banana
                        fruit_color = (0, 255, 255)  # Yellow in BGR
                        
                        # Draw curved banana shape
                        pts = np.array([
                            [center[0]-radius, center[1]],
                            [center[0]-radius//2, center[1]-radius//2],
                            [center[0]+radius//2, center[1]-radius//2],
                            [center[0]+radius, center[1]]
                        ], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(img, [pts], fruit_color)
                    
                    else:  # Pineapple
                        fruit_color = (0, 200, 255)  # Yellow-orange in BGR
                        cv2.circle(img, center, radius, fruit_color, -1)
                        
                        # Add texture
                        for i in range(0, 360, 20):
                            rad = i * math.pi / 180
                            x = int(center[0] + (radius-5) * math.cos(rad))
                            y = int(center[1] + (radius-5) * math.sin(rad))
                            cv2.circle(img, (x, y), 3, (0, 100, 200), -1)
                        
                        # Green top
                        pts = np.array([
                            [center[0]-radius//3, center[1]-radius],
                            [center[0], center[1]-radius-10],
                            [center[0]+radius//3, center[1]-radius]
                        ], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(img, [pts], (30, 180, 30))
                    
                    # Add highlight to all fruits
                    highlight_center = (
                        int(center[0] - radius * 0.3), 
                        int(center[1] - radius * 0.3)
                    )
                    cv2.circle(img, highlight_center, int(radius * 0.2), 
                             (255, 255, 255), -1)
        
        # Draw semi-transparent HUD overlay
        hud_overlay = np.zeros((100, GAME_WIDTH, 4), dtype=np.uint8)  # RGBA
        
        # Game timer
        timer_text = f"Time: {self.time_remaining}s"
        cv2.putText(hud_overlay, timer_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
        
        # Score
        score_text = f"Score: {self.score}"
        cv2.putText(hud_overlay, score_text, (GAME_WIDTH - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
        
        # Difficulty
        difficulty_text = f"Difficulty: {self.difficulty}"
        cv2.putText(hud_overlay, difficulty_text, (GAME_WIDTH // 2 - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
        
        # Combo display
        if self.combo > 1:
            combo_text = f"Combo: x{self.combo}"
            cv2.putText(img, combo_text, (GAME_WIDTH // 2 - 60, GAME_HEIGHT - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        
        # Draw HUD with transparency
        cv2.addWeighted(hud_overlay[:, :, :3], 0.7, img[:100, :], 0.3, 0, img[:100, :])
        
        # Penalties display
        if self.penalties > 0:
            penalties_text = f"Penalties: -{self.penalties}"
            cv2.putText(img, penalties_text, (20, GAME_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        
        # Game over screen
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            if self.time_remaining <= 0:
                cv2.putText(img, "TIME'S UP!", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            else:
                cv2.putText(img, "GAME OVER", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH//2 - 120, GAME_HEIGHT//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH//2 - 100, GAME_HEIGHT//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Max Combo: x{self.max_combo}", (GAME_WIDTH//2 - 100, GAME_HEIGHT//2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        return None
    
    def _create_gradient_background(self):
        """Create a gradient background as fallback"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        
        # Create a black to dark blue gradient background
        for y in range(GAME_HEIGHT):
            blue_value = int(100 * (y / GAME_HEIGHT))
            cv2.line(img, (0, y), (GAME_WIDTH, y), (blue_value, 0, 0), 1)
        
        return img