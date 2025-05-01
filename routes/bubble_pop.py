# games/bubble_pop.py

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
BUBBLE_SIZE_MIN = 50
BUBBLE_SIZE_MAX = 120
GAME_DURATION = 60  # 60 seconds

# Difficulty levels determine bubble lifetime and penalties
DIFFICULTY_LEVELS = {
    "EASY": {
        "bubble_lifetime": 8.0,  # seconds
        "score_penalty": 1,
        "spawn_rate": (2.0, 4.0)  # min and max time between spawns
    },
    "MEDIUM": {
        "bubble_lifetime": 5.0,
        "score_penalty": 2,
        "spawn_rate": (1.0, 3.0)
    },
    "HARD": {
        "bubble_lifetime": 3.0,
        "score_penalty": 3,
        "spawn_rate": (0.5, 2.0)
    }
}

class Bubble:
    def __init__(self, id, x, y, size, created_at, lifetime):
        self.id = id
        self.x = x
        self.y = y
        self.size = size
        self.popped = False
        self.popped_by_player = False  # Track if popped by player or timed out
        self.age = 0  # Track how long the bubble has existed visually for animation
        self.created_at = created_at  # When the bubble was created
        self.lifetime = lifetime  # How long the bubble lives before self-popping

class BubblePopGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id  # Track which child is playing
        if difficulty not in DIFFICULTY_LEVELS:
            self.difficulty = "MEDIUM"  # Default to medium if invalid
        
        # Game state
        self.score = 0
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        self.time_remaining = GAME_DURATION
        self.bubbles = []
        self.bubble_id_counter = 0
        self.last_bubble_spawn = datetime.now()
        self.penalties = 0  # Track total penalties
        
        # Hand tracking
        self.left_hand = None
        self.right_hand = None
        
        # Camera frame storage for AR overlay
        self.current_camera_frame = None
        
        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps

    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Bubble Pop game id: {self.game_id} with difficulty: {self.difficulty}")
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.last_bubble_spawn = datetime.now()
        self.score = 0
        self.penalties = 0
        self.time_remaining = GAME_DURATION
        self.bubbles = []
        self.bubble_id_counter = 0
    
    def update_hands(self, left_hand, right_hand):
        """Update hand positions based on hand tracking data"""
        self.left_hand = left_hand
        self.right_hand = right_hand
    
    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame
    
    def spawn_bubble(self):
        """Spawn a new bubble at a random position"""
        lifetime = DIFFICULTY_LEVELS[self.difficulty]["bubble_lifetime"]
        new_bubble = Bubble(
            id=self.bubble_id_counter,
            x=random.uniform(0, GAME_WIDTH - BUBBLE_SIZE_MAX),
            y=random.uniform(0, GAME_HEIGHT - GAME_HEIGHT * 0.15 - BUBBLE_SIZE_MAX),  # Avoid HUD area at top
            size=random.uniform(BUBBLE_SIZE_MIN, BUBBLE_SIZE_MAX),
            created_at=datetime.now(),
            lifetime=lifetime
        )
        self.bubbles.append(new_bubble)
        self.bubble_id_counter += 1
        print(f"Spawned bubble {self.bubble_id_counter} with lifetime {lifetime}s, total bubbles: {len(self.bubbles)}")
    
    def check_bubble_collisions(self):
        """Check if any hands have collided with any bubbles"""
        if not self.left_hand and not self.right_hand:
            return
        
        popped_bubbles = []
        
        for bubble in self.bubbles:
            if bubble.popped:
                continue
            
            # Scale hand coordinates to game canvas dimensions
            scale_x = GAME_WIDTH / 640
            scale_y = GAME_HEIGHT / 480
            
            # Check collision with left hand using palm center
            if self.left_hand:
                # Use palm center if available, otherwise use position
                if "palm_center" in self.left_hand:
                    hand_x = self.left_hand["palm_center"]["x"] * scale_x
                    hand_y = self.left_hand["palm_center"]["y"] * scale_y
                else:
                    hand_x = self.left_hand["position"]["x"] * scale_x
                    hand_y = self.left_hand["position"]["y"] * scale_y
                
                # Check if palm is within bubble boundaries
                bubble_center_x = bubble.x + bubble.size / 2
                bubble_center_y = bubble.y + bubble.size / 2
                
                # Calculate distance from palm center to bubble center
                distance = math.sqrt((hand_x - bubble_center_x)**2 + (hand_y - bubble_center_y)**2)
                
                # Define palm collision radius (slightly larger than single point)
                palm_radius = 20  # Adjust this value to make palm collisions more forgiving
                
                # Check if palm is within bubble (including palm radius)
                if distance < (bubble.size / 2) + palm_radius:
                    bubble.popped = True
                    bubble.popped_by_player = True
                    popped_bubbles.append(bubble)
                    print(f"Popped bubble with left palm at ({hand_x}, {hand_y})")
            
            # Check collision with right hand using palm center
            if self.right_hand and not bubble.popped:
                # Use palm center if available, otherwise use position
                if "palm_center" in self.right_hand:
                    hand_x = self.right_hand["palm_center"]["x"] * scale_x
                    hand_y = self.right_hand["palm_center"]["y"] * scale_y
                else:
                    hand_x = self.right_hand["position"]["x"] * scale_x
                    hand_y = self.right_hand["position"]["y"] * scale_y
                
                # Check if palm is within bubble boundaries
                bubble_center_x = bubble.x + bubble.size / 2
                bubble_center_y = bubble.y + bubble.size / 2
                
                # Calculate distance from palm center to bubble center
                distance = math.sqrt((hand_x - bubble_center_x)**2 + (hand_y - bubble_center_y)**2)
                
                # Define palm collision radius (slightly larger than single point)
                palm_radius = 20  # Adjust this value to make palm collisions more forgiving
                
                # Check if palm is within bubble (including palm radius)
                if distance < (bubble.size / 2) + palm_radius:
                    bubble.popped = True
                    bubble.popped_by_player = True
                    popped_bubbles.append(bubble)
                    print(f"Popped bubble with right palm at ({hand_x}, {hand_y})")
        
        # Update score
        if popped_bubbles:
            points_earned = len(popped_bubbles)
            self.score += points_earned
            print(f"Score increased to {self.score}, popped {points_earned} bubbles")
        
    def check_bubble_lifetimes(self):
        """Check if any bubbles have exceeded their lifetime and should self-pop"""
        now = datetime.now()
        expired_bubbles = []
        
        for bubble in self.bubbles:
            if bubble.popped:
                continue
                
            # Check if bubble has exceeded its lifetime
            bubble_age = (now - bubble.created_at).total_seconds()
            if bubble_age >= bubble.lifetime:
                bubble.popped = True
                bubble.popped_by_player = False
                expired_bubbles.append(bubble)
        
        # Apply penalties for expired bubbles
        if expired_bubbles:
            penalty = len(expired_bubbles) * DIFFICULTY_LEVELS[self.difficulty]["score_penalty"]
            self.penalties += penalty
            self.score = max(0, self.score - penalty)
            print(f"{len(expired_bubbles)} bubbles expired! Penalty: -{penalty}, New score: {self.score}")
    
    def clean_bubbles(self):
        """Remove popped bubbles after a delay"""
        initial_count = len(self.bubbles)
        new_bubbles = []
        for bubble in self.bubbles:
            # Keep bubbles that haven't been popped, or recently popped bubbles for visual feedback
            if not bubble.popped or bubble.age < 20:  # Show popped bubble for 20 frames
                new_bubbles.append(bubble)
                bubble.age += 1
        
        self.bubbles = new_bubbles
        if initial_count != len(self.bubbles):
            print(f"Cleaned bubbles: {initial_count} -> {len(self.bubbles)}")
    
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
        
        # Spawn new bubbles at rate based on difficulty
        min_spawn_time, max_spawn_time = DIFFICULTY_LEVELS[self.difficulty]["spawn_rate"]
        bubble_spawn_time = (now - self.last_bubble_spawn).total_seconds()
        if bubble_spawn_time > random.uniform(min_spawn_time, max_spawn_time):
            self.spawn_bubble()
            self.last_bubble_spawn = now
        
        # Check for bubble collisions with hands
        self.check_bubble_collisions()
        
        # Check for expired bubbles
        self.check_bubble_lifetimes()
        
        # Clean up popped bubbles
        self.clean_bubbles()
    
    def save_game_result(self):
        """Save the game result for reporting"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # Create game result object
            result = {
                "game_id": self.game_id,
                "difficulty": self.difficulty,
                "score": self.score,
                "duration_seconds": int(duration),
                "left_score": self.score,  # Use score as left_score for compatibility
                "right_score": self.penalties,  # Use penalties as right_score for reporting
                "timestamp": datetime.now().isoformat(),
                "child_id": self.child_id  # Include child_id in the result
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
        print(f"Rendering frame: active={self.game_active}, score={self.score}, bubbles={len(self.bubbles)}")
        
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
        overlay.fill(50)  # Dark overlay
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw bubbles
        now = datetime.now()
        for bubble in self.bubbles:
            center = (int(bubble.x + bubble.size/2), int(bubble.y + bubble.size/2))
            radius = int(bubble.size/2)
            
            if bubble.popped:
                if bubble.popped_by_player:
                    # Player-popped bubble - green fade out
                    alpha = max(0, 1 - bubble.age / 20)
                    cv2.circle(img, center, radius, (100, 255, 100, int(alpha * 128)), -1)
                else:
                    # Self-popped bubble - red fade out
                    alpha = max(0, 1 - bubble.age / 20)
                    cv2.circle(img, center, radius, (100, 100, 255, int(alpha * 128)), -1)
            else:
                # Draw bubble with gradient
                color1 = (255, 255, 255)  # White center
                color2 = (100, 200, 255)  # Light blue middle
                color3 = (50, 100, 255)   # Blue edge
                
                # Calculate remaining lifetime percentage
                bubble_age = (now - bubble.created_at).total_seconds()
                remaining_pct = max(0, 1 - (bubble_age / bubble.lifetime))
                
                # Change color based on remaining time
                if remaining_pct < 0.3:
                    # About to pop - use red tint
                    color2 = (100, 100, 255)
                    color3 = (50, 50, 255)
                elif remaining_pct < 0.6:
                    # Medium time left - use yellow tint
                    color2 = (100, 255, 255)
                    color3 = (50, 255, 255)
                
                # Draw the bubble in layers
                cv2.circle(img, center, radius, color3, -1)
                cv2.circle(img, center, int(radius * 0.8), color2, -1)
                cv2.circle(img, center, int(radius * 0.4), color1, -1)
                
                # Add a highlight spot
                highlight_center = (
                    int(center[0] - radius * 0.3), 
                    int(center[1] - radius * 0.3)
                )
                cv2.circle(img, highlight_center, int(radius * 0.2), (255, 255, 255), -1)
                
                # Draw timer fill around the bubble
                arc_end = int(360 * remaining_pct)
                cv2.ellipse(img, center, (radius + 3, radius + 3), 
                           0, 0, arc_end, (255, 255, 255), 2)
        
        # Draw hand indicators if detected
        # Draw hand indicators using palm center
        if self.left_hand or self.right_hand:
            scale_x = GAME_WIDTH / 640
            scale_y = GAME_HEIGHT / 480
            
            if self.left_hand:
                # Use palm center if available, otherwise use position
                if "palm_center" in self.left_hand:
                    hand_x = int(self.left_hand["palm_center"]["x"] * scale_x)
                    hand_y = int(self.left_hand["palm_center"]["y"] * scale_y)
                else:
                    hand_x = int(self.left_hand["position"]["x"] * scale_x)
                    hand_y = int(self.left_hand["position"]["y"] * scale_y)
                
                # Draw a glow effect around palm
                cv2.circle(img, (hand_x, hand_y), 25, (100, 200, 255), -1)
                cv2.circle(img, (hand_x, hand_y), 25, (255, 255, 255), 3)
                cv2.circle(img, (hand_x, hand_y), 15, (0, 150, 255), -1)
                
                # Optional: Draw actual palm landmarks for debugging
                if "landmarks" in self.left_hand:
                    # Draw landmarks 0, 5, 9, 13, 17 (wrist and finger bases)
                    key_landmarks = [0, 5, 9, 13, 17]
                    for idx in key_landmarks:
                        lm_x = int(self.left_hand["landmarks"][idx]["x"] * scale_x)
                        lm_y = int(self.left_hand["landmarks"][idx]["y"] * scale_y)
                        cv2.circle(img, (lm_x, lm_y), 3, (0, 255, 255), -1)
                    # Connect the landmarks to show palm outline
                    for i in range(len(key_landmarks)):
                        p1_idx = key_landmarks[i]
                        p2_idx = key_landmarks[(i+1) % len(key_landmarks)]
                        p1_x = int(self.left_hand["landmarks"][p1_idx]["x"] * scale_x)
                        p1_y = int(self.left_hand["landmarks"][p1_idx]["y"] * scale_y)
                        p2_x = int(self.left_hand["landmarks"][p2_idx]["x"] * scale_x)
                        p2_y = int(self.left_hand["landmarks"][p2_idx]["y"] * scale_y)
                        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), 2)
            
            if self.right_hand:
                # Use palm center if available, otherwise use position
                if "palm_center" in self.right_hand:
                    hand_x = int(self.right_hand["palm_center"]["x"] * scale_x)
                    hand_y = int(self.right_hand["palm_center"]["y"] * scale_y)
                else:
                    hand_x = int(self.right_hand["position"]["x"] * scale_x)
                    hand_y = int(self.right_hand["position"]["y"] * scale_y)
                
                # Draw a glow effect around palm
                cv2.circle(img, (hand_x, hand_y), 25, (100, 255, 200), -1)
                cv2.circle(img, (hand_x, hand_y), 25, (255, 255, 255), 3)
                cv2.circle(img, (hand_x, hand_y), 15, (0, 255, 150), -1)
                
                # Optional: Draw actual palm landmarks for debugging
                if "landmarks" in self.right_hand:
                    # Draw landmarks 0, 5, 9, 13, 17 (wrist and finger bases)
                    key_landmarks = [0, 5, 9, 13, 17]
                    for idx in key_landmarks:
                        lm_x = int(self.right_hand["landmarks"][idx]["x"] * scale_x)
                        lm_y = int(self.right_hand["landmarks"][idx]["y"] * scale_y)
                        cv2.circle(img, (lm_x, lm_y), 3, (0, 255, 255), -1)
                    # Connect the landmarks to show palm outline
                    for i in range(len(key_landmarks)):
                        p1_idx = key_landmarks[i]
                        p2_idx = key_landmarks[(i+1) % len(key_landmarks)]
                        p1_x = int(self.right_hand["landmarks"][p1_idx]["x"] * scale_x)
                        p1_y = int(self.right_hand["landmarks"][p1_idx]["y"] * scale_y)
                        p2_x = int(self.right_hand["landmarks"][p2_idx]["x"] * scale_x)
                        p2_y = int(self.right_hand["landmarks"][p2_idx]["y"] * scale_y)
                        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), 2)
        
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
            
            cv2.putText(img, "GAME OVER", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH//2 - 120, GAME_HEIGHT//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH//2 - 100, GAME_HEIGHT//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Penalties: {self.penalties}", (GAME_WIDTH//2 - 80, GAME_HEIGHT//2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
        
        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        return None
    
    def _create_gradient_background(self):
        """Create a gradient background as fallback"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        
        # Create a gradient blue background
        for y in range(GAME_HEIGHT):
            blue_value = int(200 - (y / GAME_HEIGHT) * 100)
            cv2.line(img, (0, y), (GAME_WIDTH, y), (blue_value, 150, 255), 1)
        
        return img