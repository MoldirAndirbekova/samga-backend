# games/letter_tracing.py

import cv2
import numpy as np
import time
import base64
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600

class LetterTracingGameState:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id
        self.game_width = 800  # Default values, will be updated
        self.game_height = 600
        
        # Game settings based on difficulty
        self.difficulty_settings = {
            "EASY": {"thickness": 45, "completion_threshold": 0.85},
            "MEDIUM": {"thickness": 35, "completion_threshold": 0.90},
            "HARD": {"thickness": 25, "completion_threshold": 0.95}
        }
        self.letter_thickness = self.difficulty_settings[difficulty]["thickness"]
        self.completion_threshold = self.difficulty_settings[difficulty]["completion_threshold"]
        
        # Game state
        self.current_letter = 'A'
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.current_letter_index = 0
        self.game_active = False
        self.game_over = False
        self.start_time = None
        self.last_update = None
        
        # Words for each letter
        self.letter_words = {
            'A': 'Apple', 'B': 'Ball', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant',
            'F': 'Fish', 'G': 'Giraffe', 'H': 'House', 'I': 'Ice', 'J': 'Jump',
            'K': 'Kite', 'L': 'Lion', 'M': 'Monkey', 'N': 'Nest', 'O': 'Orange',
            'P': 'Panda', 'Q': 'Queen', 'R': 'Rainbow', 'S': 'Sun', 'T': 'Tree',
            'U': 'Umbrella', 'V': 'Violin', 'W': 'Water', 'X': 'Xylophone',
            'Y': 'Yellow', 'Z': 'Zebra'
        }
        
        # Drawing state
        self.drawing_points = []
        self.fill_progress = 0
        self.letters_completed = 0
        self.show_congrats = False
        self.congrats_time = 0
        
        # Hand tracking
        self.current_hand = None
        
        # Camera frame for AR overlay
        self.current_camera_frame = None
        
        # Create initial letter template
        self.letter_template = None
        self.letter_mask = None
        self.create_letter_template()
        
        # FPS control
        self.fps = 30
        self.frame_time = 1 / self.fps
    
    def start_game(self, screen_width=None, screen_height=None):
        """Start or restart the game with dynamic dimensions"""
        # Update dimensions if provided
        if screen_width:
            self.game_width = screen_width
        if screen_height:
            self.game_height = screen_height
        
        print(f"Starting Letter Tracing game with dimensions: {self.game_width}x{self.game_height}")
        
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.current_letter_index = 0
        self.current_letter = self.letters[0]
        self.letters_completed = 0
        self.drawing_points = []
        self.fill_progress = 0
        self.show_congrats = False
        self.create_letter_template()
    
    def update_hand(self, hand_data):
        """Update hand position"""
        self.current_hand = hand_data
    
    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame
    
    def create_letter_template(self):
        """Create the letter template and mask with dynamic dimensions"""
        self.letter_template = np.zeros((self.game_height, self.game_width, 3), dtype=np.uint8)
        self.letter_template.fill(255)  # White background
        
        self.letter_mask = np.zeros((self.game_height, self.game_width), dtype=np.uint8)
        
        # Calculate center position
        center_x = self.game_width // 2
        center_y = self.game_height // 2
        
        # Draw the letter
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # MODIFIED: Calculate a more reasonable font scale based on screen dimensions
        # Divide by a larger number to make letters smaller
        # Use self.game_width/height instead of constants
        font_scale = min(self.game_width, self.game_height) / 40  # Changed from 25 to 40
        
        # For very large screens, cap the font size
        if font_scale > 20:
            font_scale = 20
        
        # Get text size to center it properly
        (text_width, text_height), _ = cv2.getTextSize(self.current_letter, font, font_scale, self.letter_thickness)
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        
        # Draw white outline
        border_thickness = self.letter_thickness + 6
        cv2.putText(self.letter_template, self.current_letter,
                    (text_x, text_y), font, font_scale, (255, 255, 255), border_thickness, cv2.LINE_AA)
        
        # Draw black inside letter
        cv2.putText(self.letter_template, self.current_letter,
                    (text_x, text_y), font, font_scale, (0, 0, 0), self.letter_thickness, cv2.LINE_AA)
        
        # Create mask for the letter
        cv2.putText(self.letter_mask, self.current_letter,
                    (text_x, text_y), font, font_scale, 255, self.letter_thickness, cv2.LINE_AA)
    
    def update_game_state(self):
        """Update the game state"""
        if not self.game_active or self.game_over:
            return
        
        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now
        
        # Process hand tracking and drawing - specifically use index finger tip
        if self.current_hand and "index_finger_tip" in self.current_hand:
            # Use index finger tip for letter tracing
            finger_x = int(self.current_hand["index_finger_tip"]["x"] * GAME_WIDTH / 640)
            finger_y = int(self.current_hand["index_finger_tip"]["y"] * GAME_HEIGHT / 480)
            
            # Ensure coordinates are within bounds
            finger_x = max(0, min(finger_x, GAME_WIDTH - 1))
            finger_y = max(0, min(finger_y, GAME_HEIGHT - 1))
            
            # Check if within letter bounds
            if self.letter_mask[finger_y, finger_x] > 0:
                self.drawing_points.append((finger_x, finger_y))
                self.update_fill_progress()
        
        # Check for congratulations timeout
        if self.show_congrats and time.time() - self.congrats_time > 3:
            self.next_letter()
    
    def update_fill_progress(self):
        """Update the fill progress based on drawn path"""
        if len(self.drawing_points) < 2:
            return
        
        # Create a temporary mask for the drawn path
        path_mask = np.zeros((GAME_HEIGHT, GAME_WIDTH), dtype=np.uint8)
        
        # Draw the path
        points = np.array(self.drawing_points, dtype=np.int32)
        cv2.polylines(path_mask, [points], False, 255, self.letter_thickness)
        
        # Find intersection between path and letter
        intersection = cv2.bitwise_and(path_mask, self.letter_mask)
        
        # Calculate fill progress
        total_letter_pixels = np.count_nonzero(self.letter_mask)
        filled_pixels = np.count_nonzero(intersection)
        self.fill_progress = filled_pixels / total_letter_pixels if total_letter_pixels > 0 else 0
        
        # Check if letter is completed
        if self.fill_progress >= self.completion_threshold and not self.show_congrats:
            self.show_congrats = True
            self.congrats_time = time.time()
            self.letters_completed += 1
    
    def next_letter(self):
        """Move to the next letter"""
        self.current_letter_index = (self.current_letter_index + 1) % len(self.letters)
        self.current_letter = self.letters[self.current_letter_index]
        
        # Reset drawing state
        self.drawing_points = []
        self.fill_progress = 0
        self.show_congrats = False
        
        # Create new template
        self.create_letter_template()
        
        # Check if game is complete
        if self.current_letter_index == 0 and self.letters_completed > 0:
            self.game_over = True
            self.game_active = False
            self.save_game_result()
    
    def save_game_result(self):
        """Save the game result"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate skill metrics
            skills = self.calculate_skill_metrics()
            
            # Create game result object
            result = {
                "game_id": self.game_id,
                "difficulty": self.difficulty,
                "score": self.letters_completed,
                "duration_seconds": int(duration),
                "left_score": self.letters_completed,
                "right_score": len(self.letters) - self.letters_completed,
                "timestamp": datetime.now().isoformat(),
                "skills": skills,
                "child_id": self.child_id
            }
            
            # Add to global results list
            from .games import game_results
            game_results.append(result)
            
            # Keep only the last 100 results
            if len(game_results) > 100:
                game_results.pop(0)
            
            # Persist to database
            asyncio.create_task(self._persist_to_database(result, skills))
    
    def calculate_skill_metrics(self):
        """Calculate skill metrics based on performance"""
        metrics = {}
        
        # Hand-eye coordination: Based on accuracy
        metrics["hand_eye_coordination"] = self.fill_progress * 100
        
        # Focus: Based on completion rate
        completion_rate = self.letters_completed / len(self.letters)
        metrics["focus"] = min(100, completion_rate * 100)
        
        # Fine motor skills: Based on drawing precision
        metrics["fine_motor_skills"] = min(100, (self.fill_progress * 0.8 + completion_rate * 0.2) * 100)
        
        # Perseverance: Based on number of letters attempted
        metrics["perseverance"] = min(100, (self.current_letter_index / len(self.letters)) * 100)
        
        return metrics
    
    async def _persist_to_database(self, result, skill_metrics):
        """Persist game result to database"""
        try:
            from .games import prisma
            
            if not prisma.is_connected():
                await prisma.connect()
            
            game_type_id = "letter-tracing"
            
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
            
            print(f"Saved letter tracing game report to database with ID: {game_report.id}")
            
        except Exception as e:
            print(f"Error saving letter tracing game report to database: {str(e)}")
    
    def render_frame(self):
        """Render the current game state"""
        # Start with camera frame as background if available
        if self.current_camera_frame is not None:
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
                    img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        else:
            img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        
        # Apply darkness overlay
        overlay = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Draw letter template with transparency
        letter_area = np.zeros_like(img)
        letter_area[self.letter_mask > 0] = self.letter_template[self.letter_mask > 0]
        alpha = 0.3
        cv2.addWeighted(letter_area, alpha, img, 1 - alpha, 0, img)
        
        # Draw traced path
        if len(self.drawing_points) > 1:
            points = np.array(self.drawing_points, dtype=np.int32)
            cv2.polylines(img, [points], False, (0, 0, 255), self.letter_thickness)
        
        # Draw hand indicator
        if self.current_hand and "index_finger_tip" in self.current_hand:
                # Use index finger tip position
                finger_x = int(self.current_hand["index_finger_tip"]["x"] * GAME_WIDTH / 640)
                finger_y = int(self.current_hand["index_finger_tip"]["y"] * GAME_HEIGHT / 480)
                
                # Ensure coordinates are within bounds
                finger_x = max(0, min(finger_x, GAME_WIDTH - 1))
                finger_y = max(0, min(finger_y, GAME_HEIGHT - 1))
                
                # Draw index finger indicator with smaller size
                cv2.circle(img, (finger_x, finger_y), 15, (0, 255, 0), -1)  # Smaller circle for finger
                cv2.circle(img, (finger_x, finger_y), 15, (255, 255, 255), 2)
                cv2.circle(img, (finger_x, finger_y), 10, (0, 200, 0), -1)  # Inner circle for finger tip
                
                # Optionally draw the finger landmark trace for visual feedback
                if "landmarks" in self.current_hand:
                    # Draw line from wrist to index finger tip
                    wrist = self.current_hand["landmarks"][0]
                    if hasattr(wrist, '__getitem__'):
                        wrist_x = int(wrist["x"] * GAME_WIDTH / 640)
                        wrist_y = int(wrist["y"] * GAME_HEIGHT / 480)
                        cv2.line(img, (wrist_x, wrist_y), (finger_x, finger_y), (0, 255, 0, 128), 2)
        
        # Draw UI elements
        cv2.putText(img, "Trace the letter with your finger", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Progress: {int(self.fill_progress * 100)}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Letters completed: {self.letters_completed}/{len(self.letters)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show congratulations message
        if self.show_congrats:
            congrats_text = f"Good job! {self.letter_words[self.current_letter]} starts with {self.current_letter}!"
            text_size = cv2.getTextSize(congrats_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (GAME_WIDTH - text_size[0]) // 2
            text_y = GAME_HEIGHT - 50
            
            cv2.putText(img, congrats_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(img, congrats_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Game over screen
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            cv2.putText(img, "GAME COMPLETE!", (GAME_WIDTH//2 - 150, GAME_HEIGHT//2 - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.putText(img, f"Letters completed: {self.letters_completed}/{len(self.letters)}",
                       (GAME_WIDTH//2 - 150, GAME_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to base64
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        
        return None