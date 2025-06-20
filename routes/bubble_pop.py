# games/bubble_pop.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
import cv2
import random
import base64
import math
import os

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
BUBBLE_SIZE_MIN = 50
BUBBLE_SIZE_MAX = 120
GAME_DURATION = 60  # 60 seconds

# Paths to balloon images - you can add multiple balloon PNG files here
BALLOON_IMAGE_PATHS = [
    os.path.join("static", "images", "baloon-yellow.png"),
    os.path.join("static", "images", "baloon-red.png"),
    os.path.join("static", "images", "baloon-blue.png"),
    os.path.join("static", "images", "baloon-green.png"),
    os.path.join("static", "images", "baloon-purple.png"),
    os.path.join("static", "images", "baloon-pink.png"),
    os.path.join("static", "images", "baloon-orange.png"),
]

# Paths to popped/changed balloon images
BALLOON_CHANGED_IMAGE_PATHS = [
    os.path.join("static", "images", "yellow-changed.png"),
    os.path.join("static", "images", "red-changed.png"),
    os.path.join("static", "images", "blue-changed.png"),
    os.path.join("static", "images", "green-changed.png"),
    os.path.join("static", "images", "purple-changed.png"),
    os.path.join("static", "images", "pink-changed.png"),
    os.path.join("static", "images", "orange-changed.png"),
]

# Fallback balloon colors if PNG images aren't available
BALLOON_COLORS = [
    (255, 100, 100),  # Light Red
    (100, 255, 100),  # Light Green
    (100, 100, 255),  # Light Blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 150, 100),  # Orange
    (150, 100, 255),  # Purple
]

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
        self.color = random.choice(BALLOON_COLORS)  # Fallback color if no PNG
        self.balloon_image_index = random.randint(0, max(0, len(BALLOON_IMAGE_PATHS) - 1))  # Random balloon image
        self.pop_start_time = None  # When the balloon was popped (for timing the changed image display)
        self.bob_offset = random.uniform(0, 2 * math.pi)  # For floating animation

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
        
        # Pause tracking
        self.pause_start_time = None
        self.total_pause_time = 0
        
        # Hand tracking
        self.left_hand = None
        self.right_hand = None
        
        # Camera frame storage for AR overlay
        self.current_camera_frame = None
        
        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps
        
        # Animation frame counter
        self.animation_frame = 0
        
        # Load balloon images
        self.balloon_images = []
        self.balloon_changed_images = []
        self.load_balloon_images()
    
    def load_balloon_images(self):
        """Load balloon PNG images and their changed versions"""
        print("Loading balloon images...")
        print(f"Looking for {len(BALLOON_IMAGE_PATHS)} balloon files:")
        
        # Load normal balloon images
        for i, path in enumerate(BALLOON_IMAGE_PATHS):
            print(f"  {i+1}. {path}")
            try:
                balloon_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if balloon_img is not None:
                    print(f"    ✅ Successfully loaded! Shape: {balloon_img.shape}")
                    
                    # Check if image has alpha channel, if not add one
                    if len(balloon_img.shape) == 3 and balloon_img.shape[2] == 3:
                        # Convert BGR to BGRA
                        alpha = np.ones((balloon_img.shape[0], balloon_img.shape[1]), dtype=balloon_img.dtype) * 255
                        balloon_img = cv2.merge((balloon_img, alpha))
                        print(f"    Added alpha channel, new shape: {balloon_img.shape}")
                    
                    self.balloon_images.append(balloon_img)
                else:
                    print(f"    ❌ Could not load (file exists but cv2.imread failed)")
                    self.balloon_images.append(None)
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                self.balloon_images.append(None)
        
        # Load changed balloon images
        print(f"Looking for {len(BALLOON_CHANGED_IMAGE_PATHS)} changed balloon files:")
        for i, path in enumerate(BALLOON_CHANGED_IMAGE_PATHS):
            print(f"  {i+1}. {path}")
            try:
                changed_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if changed_img is not None:
                    print(f"    ✅ Successfully loaded! Shape: {changed_img.shape}")
                    
                    # Check if image has alpha channel, if not add one
                    if len(changed_img.shape) == 3 and changed_img.shape[2] == 3:
                        # Convert BGR to BGRA
                        alpha = np.ones((changed_img.shape[0], changed_img.shape[1]), dtype=changed_img.dtype) * 255
                        changed_img = cv2.merge((changed_img, alpha))
                        print(f"    Added alpha channel, new shape: {changed_img.shape}")
                    
                    self.balloon_changed_images.append(changed_img)
                else:
                    print(f"    ❌ Could not load (file exists but cv2.imread failed)")
                    self.balloon_changed_images.append(None)
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                self.balloon_changed_images.append(None)
        
        if not self.balloon_images or all(img is None for img in self.balloon_images):
            print("❌ No balloon PNG images loaded - will use programmatic balloon generation")
        else:
            print(f"✅ Successfully loaded {len([img for img in self.balloon_images if img is not None])} balloon images")
        
        if not self.balloon_changed_images or all(img is None for img in self.balloon_changed_images):
            print("❌ No changed balloon PNG images loaded - will use programmatic pop effects")
        else:
            print(f"✅ Successfully loaded {len([img for img in self.balloon_changed_images if img is not None])} changed balloon images")
        
        print("=" * 50)

    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Balloon Pop game id: {self.game_id} with difficulty: {self.difficulty}")
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
        self.pause_start_time = None
        self.total_pause_time = 0
        
        # Ensure balloon images are loaded
        if not self.balloon_images or not self.balloon_changed_images:
            self.load_balloon_images()
    
    def pause_game(self):
        """Pause the game"""
        print(f"Pausing Bubble Pop game id: {self.game_id}")
        if self.game_active and not self.game_over:
            self.game_active = False
            self.pause_start_time = datetime.now()
    
    def resume_game(self):
        """Resume the game"""
        print(f"Resuming Bubble Pop game id: {self.game_id}")
        if not self.game_active and not self.game_over and self.pause_start_time:
            self.game_active = True
            # Track total pause time
            pause_duration = (datetime.now() - self.pause_start_time).total_seconds()
            self.total_pause_time += pause_duration
            self.pause_start_time = None
            # Reset the last update time to prevent time jumps
            self.last_update = datetime.now()
            # Also reset bubble spawn time to prevent immediate spawning
            self.last_bubble_spawn = datetime.now()
    
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
        print(f"Spawned balloon {self.bubble_id_counter} with lifetime {lifetime}s, total balloons: {len(self.bubbles)}")
    
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
                    bubble.pop_start_time = datetime.now()
                    popped_bubbles.append(bubble)
                    print(f"Popped balloon with left palm at ({hand_x}, {hand_y})")
            
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
                    bubble.pop_start_time = datetime.now()
                    popped_bubbles.append(bubble)
                    print(f"Popped balloon with right palm at ({hand_x}, {hand_y})")
        
        # Update score
        if popped_bubbles:
            points_earned = len(popped_bubbles)
            self.score += points_earned
            print(f"Score increased to {self.score}, popped {points_earned} balloons")
        
    def check_bubble_lifetimes(self):
        """Check if any bubbles have exceeded their lifetime and should pop automatically"""
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
                bubble.pop_start_time = datetime.now()
                expired_bubbles.append(bubble)
        
        # Apply penalties for expired bubbles
        if expired_bubbles:
            penalty = len(expired_bubbles) * DIFFICULTY_LEVELS[self.difficulty]["score_penalty"]
            self.penalties += penalty
            self.score = max(0, self.score - penalty)
            print(f"{len(expired_bubbles)} balloons expired! Penalty: -{penalty}, New score: {self.score}")
    
    def clean_bubbles(self):
        """Remove popped bubbles after showing changed image for 3 seconds"""
        initial_count = len(self.bubbles)
        new_bubbles = []
        current_time = datetime.now()
        
        for bubble in self.bubbles:
            if bubble.popped and bubble.pop_start_time:
                # Show changed image for 3 seconds, then remove
                time_since_pop = (current_time - bubble.pop_start_time).total_seconds()
                if time_since_pop < 3.0:  # Show for 3 seconds
                    new_bubbles.append(bubble)
                # Otherwise, remove the bubble (don't add to new_bubbles)
            elif not bubble.popped:
                new_bubbles.append(bubble)
                bubble.age += 1
        
        self.bubbles = new_bubbles
        if initial_count != len(self.bubbles):
            print(f"Cleaned balloons: {initial_count} -> {len(self.bubbles)}")
    
    def update_game_state(self):
        """Update the game state for one frame"""
        if not self.game_active or self.game_over:
            return
        
        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now
        
        # Update game timer (excluding pause time)
        elapsed_seconds = (now - self.start_time).total_seconds() - self.total_pause_time
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
            duration = (datetime.now() - self.start_time).total_seconds() - self.total_pause_time
            
            # Create game result object
            result = {
                "game_id": self.game_id,
                "game_name": "Balloon Pop",
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
    
    def create_balloon_shape(self, size, color):
        """Create a balloon shape programmatically as fallback"""
        balloon_img = np.zeros((int(size * 1.4), int(size), 4), dtype=np.uint8)
        
        # Main balloon body (oval)
        center_x, center_y = int(size / 2), int(size * 0.4)
        balloon_width, balloon_height = int(size * 0.8), int(size * 0.7)
        
        # Create balloon gradient
        for y in range(balloon_img.shape[0]):
            for x in range(balloon_img.shape[1]):
                # Check if point is inside oval
                dx = (x - center_x) / (balloon_width / 2)
                dy = (y - center_y) / (balloon_height / 2)
                
                if dx*dx + dy*dy <= 1:
                    # Create gradient effect
                    highlight_factor = max(0, 1 - (dx*dx + dy*dy))
                    
                    # Add highlight on the upper left
                    highlight_x = x - center_x + balloon_width * 0.2
                    highlight_y = y - center_y + balloon_height * 0.2
                    highlight_dist = (highlight_x*highlight_x + highlight_y*highlight_y) / (balloon_width * balloon_height * 0.1)
                    highlight = max(0, 1 - highlight_dist) * 0.5
                    
                    # Apply color with highlight
                    balloon_img[y, x, 0] = min(255, color[0] + highlight * 100)  # B
                    balloon_img[y, x, 1] = min(255, color[1] + highlight * 100)  # G
                    balloon_img[y, x, 2] = min(255, color[2] + highlight * 100)  # R
                    balloon_img[y, x, 3] = 255  # Full opacity - no fading
        
        # Add balloon string
        string_start_y = int(size * 0.75)
        string_end_y = int(size * 1.2)
        string_x = center_x
        
        for y in range(string_start_y, min(string_end_y, balloon_img.shape[0])):
            # Slightly wavy string
            wave = int(2 * math.sin(y * 0.3))
            string_pos_x = string_x + wave
            if 0 <= string_pos_x < balloon_img.shape[1]:
                balloon_img[y, string_pos_x, :3] = [50, 50, 50]  # Dark gray string
                balloon_img[y, string_pos_x, 3] = 255
        
        return balloon_img
    
    def create_popped_balloon_shape(self, size, color):
        """Create a popped balloon shape programmatically as fallback"""
        balloon_img = np.zeros((int(size * 1.4), int(size), 4), dtype=np.uint8)
        
        # Create burst/explosion effect
        center_x, center_y = int(size / 2), int(size * 0.4)
        
        # Draw irregular burst lines
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            for length in range(5, int(size * 0.6), 3):
                # Add some randomness to the burst lines
                noise_x = random.randint(-3, 3)
                noise_y = random.randint(-3, 3)
                
                end_x = int(center_x + math.cos(rad) * length + noise_x)
                end_y = int(center_y + math.sin(rad) * length + noise_y)
                
                if 0 <= end_x < balloon_img.shape[1] and 0 <= end_y < balloon_img.shape[0]:
                    # Full intensity - no fading
                    balloon_img[end_y, end_x, 0] = color[0]  # B
                    balloon_img[end_y, end_x, 1] = color[1]  # G
                    balloon_img[end_y, end_x, 2] = color[2]  # R
                    balloon_img[end_y, end_x, 3] = 255  # Full opacity
        
        # Add some scattered particles
        for _ in range(20):
            particle_x = random.randint(0, balloon_img.shape[1] - 1)
            particle_y = random.randint(0, balloon_img.shape[0] - 1)
            
            # Distance from center affects intensity
            dist_from_center = math.sqrt((particle_x - center_x)**2 + (particle_y - center_y)**2)
            if dist_from_center < size * 0.8:
                balloon_img[particle_y, particle_x, 0] = color[0]
                balloon_img[particle_y, particle_x, 1] = color[1]
                balloon_img[particle_y, particle_x, 2] = color[2]
                balloon_img[particle_y, particle_x, 3] = 255
        
        return balloon_img
    
    def overlay_balloon_image(self, img, bubble):
        """Overlay the balloon PNG image or created balloon on the game frame"""
        # Calculate floating animation (only for non-popped balloons)
        if not bubble.popped:
            bob_amplitude = 3
            bob_y = bubble.y + bob_amplitude * math.sin((self.animation_frame * 0.1) + bubble.bob_offset)
        else:
            bob_y = bubble.y  # No floating animation when popped
        
        # Calculate the size to display the balloon
        balloon_width = int(bubble.size)
        balloon_height = int(bubble.size * 1.2)  # Slightly taller than wide for balloon shape
        
        # Calculate position to place the balloon
        x_pos = int(bubble.x)
        y_pos = int(bob_y)
        
        # Make sure we're within bounds
        if (x_pos + balloon_width > img.shape[1] or 
            y_pos + balloon_height > img.shape[0] or
            x_pos < 0 or y_pos < 0):
            return
        
        # Choose the appropriate image based on bubble state
        if bubble.popped:
            # Use changed/popped image
            if (self.balloon_changed_images and 
                bubble.balloon_image_index < len(self.balloon_changed_images) and
                self.balloon_changed_images[bubble.balloon_image_index] is not None):
                balloon_image = self.balloon_changed_images[bubble.balloon_image_index]
                resized_balloon = cv2.resize(balloon_image, (balloon_width, balloon_height))
            else:
                # Fallback to programmatic popped balloon
                resized_balloon = self.create_popped_balloon_shape(bubble.size, bubble.color)
                resized_balloon = cv2.resize(resized_balloon, (balloon_width, balloon_height))
        else:
            # Use normal balloon image
            if (self.balloon_images and 
                bubble.balloon_image_index < len(self.balloon_images) and
                self.balloon_images[bubble.balloon_image_index] is not None):
                balloon_image = self.balloon_images[bubble.balloon_image_index]
                resized_balloon = cv2.resize(balloon_image, (balloon_width, balloon_height))
            else:
                # Fallback to programmatic balloon
                resized_balloon = self.create_balloon_shape(bubble.size, bubble.color)
                resized_balloon = cv2.resize(resized_balloon, (balloon_width, balloon_height))
        
        # Apply the balloon image to the frame
        if len(resized_balloon.shape) == 3 and resized_balloon.shape[2] == 4:
            # Extract alpha channel - no fading effects, just use original alpha
            alpha = resized_balloon[:, :, 3] / 255.0
            
            # Calculate region for overlay
            y1, y2 = y_pos, y_pos + balloon_height
            x1, x2 = x_pos, x_pos + balloon_width
            
            # Make sure region is within image boundaries
            if y1 >= 0 and y2 <= img.shape[0] and x1 >= 0 and x2 <= img.shape[1]:
                # Get the region of the background
                bg_region = img[y1:y2, x1:x2]
                
                # For each color channel
                for c in range(3):
                    # Overlay the balloon onto the image
                    img[y1:y2, x1:x2, c] = (1 - alpha) * bg_region[:, :, c] + alpha * resized_balloon[:, :, c]
        else:
            # No alpha channel, simple overlay - no fading effects
            y1, y2 = y_pos, y_pos + balloon_height
            x1, x2 = x_pos, x_pos + balloon_width
            
            if y1 >= 0 and y2 <= img.shape[0] and x1 >= 0 and x2 <= img.shape[1]:
                # Direct overlay without any fading
                balloon_region = resized_balloon[:, :, :3] if resized_balloon.shape[2] > 3 else resized_balloon
                img[y1:y2, x1:x2] = balloon_region
    
    def render_frame(self):
        """Render the current game state to an image"""
        print(f"Rendering frame: active={self.game_active}, score={self.score}, balloons={len(self.bubbles)}")
        
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
        overlay[:, :] = (20, 30, 40)  # Slight blue tint
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Draw balloons using PNG images or programmatic shapes
        for bubble in self.bubbles:
            self.overlay_balloon_image(img, bubble)
        
        # Draw hand indicators if detected
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
                
                # Draw animated glow effect around palm
                glow_radius = 25 + int(5 * math.sin(self.animation_frame * 0.2))
                cv2.circle(img, (hand_x, hand_y), glow_radius, (100, 200, 255), -1)
                cv2.circle(img, (hand_x, hand_y), glow_radius, (255, 255, 255), 3)
                cv2.circle(img, (hand_x, hand_y), 15, (0, 150, 255), -1)
            
            if self.right_hand:
                # Use palm center if available, otherwise use position
                if "palm_center" in self.right_hand:
                    hand_x = int(self.right_hand["palm_center"]["x"] * scale_x)
                    hand_y = int(self.right_hand["palm_center"]["y"] * scale_y)
                else:
                    hand_x = int(self.right_hand["position"]["x"] * scale_x)
                    hand_y = int(self.right_hand["position"]["y"] * scale_y)
                
                # Draw animated glow effect around palm
                glow_radius = 25 + int(5 * math.sin(self.animation_frame * 0.2))
                cv2.circle(img, (hand_x, hand_y), glow_radius, (100, 255, 200), -1)
                cv2.circle(img, (hand_x, hand_y), glow_radius, (255, 255, 255), 3)
                cv2.circle(img, (hand_x, hand_y), 15, (0, 255, 150), -1)
        
        # Draw enhanced HUD overlay
        hud_height = 80
        hud_overlay = np.zeros((hud_height, GAME_WIDTH, 3), dtype=np.uint8)
        hud_overlay[:, :] = (0, 0, 0)  # Black background
        cv2.addWeighted(hud_overlay, 0.6, img[:hud_height, :], 0.4, 0, img[:hud_height, :])
        
        # Game timer with enhanced styling
        timer_text = f"Time: {self.time_remaining}s"
        cv2.putText(img, timer_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, timer_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        
        # Score with enhanced styling
        score_text = f"Score: {self.score}"
        cv2.putText(img, score_text, (GAME_WIDTH - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, score_text, (GAME_WIDTH - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
        
        # Difficulty with enhanced styling
        difficulty_text = f"{self.difficulty}"
        cv2.putText(img, difficulty_text, (GAME_WIDTH // 2 - 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, difficulty_text, (GAME_WIDTH // 2 - 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)
        
        # Penalties display
        if self.penalties > 0:
            penalties_text = f"Missed: {self.penalties}"
            cv2.putText(img, penalties_text, (20, GAME_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, penalties_text, (20, GAME_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 1)
        
        # Game over screen
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            
            # Game over text with glow effect
            cv2.putText(img, "GAME OVER", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.putText(img, "GAME OVER", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 100), 3)
            
            cv2.putText(img, f"Final Score: {self.score}", (GAME_WIDTH//2 - 120, GAME_HEIGHT//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Difficulty: {self.difficulty}", (GAME_WIDTH//2 - 100, GAME_HEIGHT//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(img, f"Balloons Missed: {self.penalties}", (GAME_WIDTH//2 - 120, GAME_HEIGHT//2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
        
        # Increment animation frame
        self.animation_frame += 1
        
        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        return None
    
    def _create_gradient_background(self):
        """Create a gradient background as fallback"""
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        
        # Create a sky-like gradient
        for y in range(GAME_HEIGHT):
            # Sky blue to lighter blue gradient
            blue_value = int(135 + (y / GAME_HEIGHT) * 120)
            green_value = int(206 + (y / GAME_HEIGHT) * 49)
            red_value = int(235 + (y / GAME_HEIGHT) * 20)
            
            cv2.line(img, (0, y), (GAME_WIDTH, y), (blue_value, green_value, red_value), 1)
        
        return img