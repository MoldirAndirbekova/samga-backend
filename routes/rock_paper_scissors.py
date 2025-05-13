import random
import time
import cv2
import os
import numpy as np
from datetime import datetime
import base64
import math


class RockPaperScissorsGame:
    def __init__(self, game_id, difficulty="MEDIUM", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id

        # Game setup constants
        self.GAME_WIDTH = 800
        self.GAME_HEIGHT = 600
        self.fps = 30
        self.frame_time = 1 / self.fps

        # We'll use MediaPipe hands directly instead of cvzone
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Game state variables
        self.game_active = False
        self.game_over = False
        self.score = 0
        self.computer_score = 0
        self.start_time = None
        self.last_update = None
        self.current_camera_frame = None
        self.current_state = 'HOME'
        self.round_count = 0
        self.current_round = 0
        self.countdown = 0
        self.round_result = None
        self.player_move = None
        self.computer_move = None
        self.game_background = None

        # Difficulty settings
        self.DIFFICULTY_LEVELS = {
            "EASY": {
                "rounds": 3,
                "decision_time": 3,
                "computer_randomness": 0.7  # Computer is more predictable
            },
            "MEDIUM": {
                "rounds": 5,
                "decision_time": 2,
                "computer_randomness": 0.85
            },
            "HARD": {
                "rounds": 7,
                "decision_time": 1.5,
                "computer_randomness": 1.0  # Computer is fully random
            }
        }

        # Load resources - using your existing paths
        try:
            # Try to use the resources from your original RockPaper.py
            self.resources = {
                'rock': self._load_image_with_alpha("Resources/1.png"),
                'paper': self._load_image_with_alpha("Resources/2.png"),
                'scissors': self._load_image_with_alpha("Resources/3.png")
            }

            # Use existing background images if available
            self.backgrounds = {
                'home': self._load_image_or_create_gradient("Resources/Start-exit.png"),
                'level': self._load_image_or_create_gradient("Resources/Yes-change.png"),
                'game': self._load_image_or_create_gradient("Resources/Robot-player.png"),
                'result': self._load_image_or_create_gradient("Resources/game_score_bg.png"),
                'win': self._load_image_or_create_gradient("Resources/won.png"),
                'lose': self._load_image_or_create_gradient("Resources/lost.png"),
                'draw': self._load_image_or_create_gradient("Resources/draw.png")
            }
        except Exception as e:
            print(f"Error loading resources: {e}")
            # Create placeholder for missing resources
            self._create_placeholder_resources()

        # Set initial background
        self.game_background = self.backgrounds.get('home', self._create_gradient_background())

    def _load_image_with_alpha(self, path):
        """Loads an image with alpha channel if available"""
        try:
            if os.path.exists(path):
                # Check if image has alpha channel
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:  # Has alpha channel
                    return img
                else:  # No alpha channel
                    return cv2.imread(path)
            else:
                print(f"Image not found: {path}")
                return None
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def _load_image_or_create_gradient(self, path):
        """Loads an image if it exists, otherwise creates a gradient"""
        try:
            if os.path.exists(path):
                return cv2.imread(path)
            else:
                print(f"Image not found: {path}, using gradient")
                return self._create_gradient_background()
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return self._create_gradient_background()

    def _create_gradient_background(self):
        """Creates a gradient background as fallback"""
        image = np.zeros((self.GAME_HEIGHT, self.GAME_WIDTH, 3), dtype=np.uint8)
        for y in range(self.GAME_HEIGHT):
            blue_value = int(200 - (y / self.GAME_HEIGHT) * 100)
            cv2.line(image, (0, y), (self.GAME_WIDTH, y), (blue_value, 100, 50), 1)
        return image

    def _create_placeholder_resources(self):
        """Creates placeholder resources if original resources cannot be loaded"""
        # Create empty dictionaries to avoid None errors
        self.resources = {}
        self.backgrounds = {}

        # Create placeholder images for rock, paper, scissors
        for move in ['rock', 'paper', 'scissors']:
            img = np.zeros((100, 100, 4), dtype=np.uint8)
            cv2.putText(img, move.upper(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
            self.resources[move] = img

        # Create placeholder backgrounds
        for bg_type in ['home', 'level', 'game', 'result', 'win', 'lose', 'draw']:
            self.backgrounds[bg_type] = self._create_gradient_background()

    def overlay_image_with_alpha(self, background, foreground, x, y):
        """Custom function to overlay images with alpha channel (without cvzone)"""
        if foreground is None or background is None:
            return background

        # Ensure foreground has 4 channels (RGBA)
        if foreground.shape[2] != 4:
            return background

        # Get dimensions
        fg_height, fg_width = foreground.shape[:2]
        bg_height, bg_width = background.shape[:2]

        # Check if the region is within the background image
        if x < 0 or y < 0 or x + fg_width > bg_width or y + fg_height > bg_height:
            # Adjust for partially visible image
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(bg_width, x + fg_width)
            y_end = min(bg_height, y + fg_height)

            fg_x_start = x_start - x
            fg_y_start = y_start - y
            fg_x_end = fg_x_start + (x_end - x_start)
            fg_y_end = fg_y_start + (y_end - y_start)

            roi = background[y_start:y_end, x_start:x_end]
            fg_partial = foreground[fg_y_start:fg_y_end, fg_x_start:fg_x_end]

            # Apply alpha blending
            alpha = fg_partial[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            # Create foreground RGB
            fg_rgb = fg_partial[:, :, :3]

            # Blend
            blended = roi * (1 - alpha) + fg_rgb * alpha
            background[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
        else:
            # Region fully within background
            roi = background[y:y + fg_height, x:x + fg_width]

            # Apply alpha blending
            alpha = foreground[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            # Create foreground RGB
            fg_rgb = foreground[:, :, :3]

            # Blend
            blended = roi * (1 - alpha) + fg_rgb * alpha
            background[y:y + fg_height, x:x + fg_width] = blended.astype(np.uint8)

        return background

    def find_hands(self, image):
        """Process image to find hand landmarks using MediaPipe"""
        if image is None:
            return [], image

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(image_rgb)

        hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image (for visualization)
                self._draw_landmarks(image, hand_landmarks)

                # Process landmarks to match our format
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))

                # Add hand to list
                hands.append({
                    'landmarks': landmarks,
                    'type': self._detect_hand_type(landmarks)
                })

        return hands, image

    def _draw_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on the image"""
        h, w, c = image.shape

        # Draw connection lines
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (9, 10), (10, 11), (11, 12),  # Middle finger
            (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # Palm
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(hand_landmarks.landmark[start_idx].x * w),
                           int(hand_landmarks.landmark[start_idx].y * h))
            end_point = (int(hand_landmarks.landmark[end_idx].x * w),
                         int(hand_landmarks.landmark[end_idx].y * h))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Draw landmark points
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return image

    def _detect_hand_type(self, landmarks):
        """Simple logic to detect if it's a left or right hand"""
        # This is a very simplified detection - in real applications you'd use the handedness from MediaPipe
        # For now, we'll just assume it's a right hand
        return "Right"

    def fingersUp(self, hand):
        """Check which fingers are up (simplified version of cvzone's fingersUp)"""
        if not hand or 'landmarks' not in hand:
            return [0, 0, 0, 0, 0]

        landmarks = hand['landmarks']
        fingers = []

        # Thumb (special case)
        # If the tip of the thumb is to the right of the joint, it's up for right hand
        if hand['type'] == 'Right':
            if landmarks[4][0] > landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if landmarks[4][0] < landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # For other fingers, check if the tip is above the second joint
        for tip_idx, mid_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip_idx][1] < landmarks[mid_idx][1]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

        return fingers

    def start_game(self):
        """Start or restart the game"""
        print(f"Starting Rock Paper Scissors game id: {self.game_id} with difficulty: {self.difficulty}")
        self.game_active = True
        self.game_over = False
        self.score = 0
        self.computer_score = 0
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.current_round = 0

        # Set rounds based on difficulty
        self.round_count = self.DIFFICULTY_LEVELS[self.difficulty]['rounds']

        # Move to preview state
        self.current_state = 'PREVIEW'

    def update_camera_frame(self, frame):
        """Store the current camera frame for AR overlay"""
        self.current_camera_frame = frame

    def update_hands(self, left_hand, right_hand):
        """Update hand positions based on hand tracking data"""
        # We'll use the first detected hand for simplicity
        self.hand = left_hand if left_hand else right_hand

    def finger_gesture_detection(self, fingers):
        """Identify the gesture (rock, paper, scissors) based on fingers"""
        if not fingers:
            return None

        # Rock: closed fist (0 fingers up)
        if sum(fingers) == 0:
            return "rock"
        # Paper: all fingers extended (5 fingers up)
        elif sum(fingers) == 5:
            return "paper"
        # Scissors: index and middle finger extended (2 fingers up in specific pattern)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return "scissors"
        # Other hand positions are not valid moves
        return None

    def get_computer_move(self):
        """Get computer's move - can be adjusted based on difficulty"""
        moves = ["rock", "paper", "scissors"]

        # Basic AI: If difficulty is not HARD, sometimes counter the player's last move
        if self.difficulty != "HARD" and self.player_move and random.random() > self.DIFFICULTY_LEVELS[self.difficulty][
            "computer_randomness"]:
            # Counter strategy
            if self.player_move == "rock":
                return "paper"
            elif self.player_move == "paper":
                return "scissors"
            else:  # scissors
                return "rock"

        # Otherwise random
        return random.choice(moves)

    def determine_winner(self, player_move, computer_move):
        """Determine the winner of the round"""
        if player_move == computer_move:
            return "draw"
        elif (player_move == "rock" and computer_move == "scissors") or \
                (player_move == "paper" and computer_move == "rock") or \
                (player_move == "scissors" and computer_move == "paper"):
            return "player"
        else:
            return "computer"

    def update_game_state(self):
        """Update the game state based on current state and inputs"""
        if not self.game_active and self.current_state != 'HOME':
            return

        # Calculate time delta
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now

        # State machine for game flow
        if self.current_state == 'HOME':
            # Logic for home screen - waiting for player to start
            pass

        elif self.current_state == 'PREVIEW':
            # Show preview for a few seconds then move to level selection
            if not hasattr(self, 'preview_timer'):
                self.preview_timer = 3  # 3 seconds preview

            self.preview_timer -= dt
            if self.preview_timer <= 0:
                self.current_state = 'LEVEL_SELECT'
                del self.preview_timer

        elif self.current_state == 'LEVEL_SELECT':
            # Level selection logic handled in handle_input
            pass

        elif self.current_state == 'GAME_READY':
            # Countdown before each round
            if not hasattr(self, 'ready_timer'):
                self.ready_timer = 3
                self.countdown = 3

            self.ready_timer -= dt
            self.countdown = max(0, int(self.ready_timer) + 1)

            if self.ready_timer <= 0:
                self.current_state = 'GAME_MOVE'
                self.player_move = None
                self.computer_move = None
                self.move_timer = self.DIFFICULTY_LEVELS[self.difficulty]["decision_time"]
                del self.ready_timer

        elif self.current_state == 'GAME_MOVE':
            # Player makes their move
            self.move_timer -= dt

            if self.move_timer <= 0 or self.player_move:
                # If time's up or player made a move
                if not self.player_move:
                    self.player_move = "rock"  # Default to rock if no move made

                self.computer_move = self.get_computer_move()
                self.round_result = self.determine_winner(self.player_move, self.computer_move)

                # Update scores
                if self.round_result == "player":
                    self.score += 1
                elif self.round_result == "computer":
                    self.computer_score += 1

                self.current_round += 1
                self.current_state = 'ROUND_RESULT'
                self.result_timer = 2  # Show result for 2 seconds

        elif self.current_state == 'ROUND_RESULT':
            # Show round result
            self.result_timer -= dt

            if self.result_timer <= 0:
                # Check if game is over
                if self.current_round >= self.round_count:
                    self.current_state = 'GAME_OVER'
                    self.game_over = True
                    self.save_game_result()
                else:
                    # Next round
                    self.current_state = 'GAME_READY'

        elif self.current_state == 'GAME_OVER':
            # Game over state
            if not hasattr(self, 'game_over_timer'):
                self.game_over_timer = 5  # Show game over screen for 5 seconds

            self.game_over_timer -= dt

            if self.game_over_timer <= 0:
                self.current_state = 'HOME'
                self.game_active = False
                del self.game_over_timer

    def handle_input(self, hands):
        """Process hand detection and gestures as input"""
        if not hands:
            return

        # Get the first hand
        hand = hands[0]
        fingers = self.fingersUp(hand)

        # Different input handling based on game state
        if self.current_state == 'HOME':
            # On home screen, detect a fist (rock) to start game
            gesture = self.finger_gesture_detection(fingers)
            if gesture == "rock":
                self.start_game()

        elif self.current_state == 'LEVEL_SELECT':
            # Detect number of fingers for level selection
            finger_count = sum(fingers)

            if finger_count == 1:
                self.difficulty = "EASY"
                self.round_count = self.DIFFICULTY_LEVELS["EASY"]["rounds"]
                self.current_state = 'GAME_READY'
            elif finger_count == 2:
                self.difficulty = "MEDIUM"
                self.round_count = self.DIFFICULTY_LEVELS["MEDIUM"]["rounds"]
                self.current_state = 'GAME_READY'
            elif finger_count == 3:
                self.difficulty = "HARD"
                self.round_count = self.DIFFICULTY_LEVELS["HARD"]["rounds"]
                self.current_state = 'GAME_READY'

        elif self.current_state == 'GAME_MOVE':
            # Detect rock, paper, scissors gesture
            gesture = self.finger_gesture_detection(fingers)
            if gesture:
                self.player_move = gesture

    def save_game_result(self):
        """Save game results for reporting"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()

            # Determine final result
            if self.score > self.computer_score:
                result = "win"
            elif self.score < self.computer_score:
                result = "lose"
            else:
                result = "draw"

            # Create result object
            game_result = {
                "game_id": self.game_id,
                "game_name": "Rock Paper Scissors",
                "difficulty": self.difficulty,
                "score": self.score,
                "computer_score": self.computer_score,
                "duration_seconds": int(duration),
                "rounds_played": self.current_round,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "child_id": self.child_id
            }

            print(f"Game Result: {game_result}")

            # This would typically be sent to a database or API
            try:
                # Import here to avoid circular import issues
                from .games import game_results
                game_results.append(game_result)
                print(f"Saved game result: {game_result}")
            except ImportError:
                print("Could not save game result - games module not available")

    def render_frame(self):
        """Render the current game state to an image"""
        # Start with camera frame as background if available
        if self.current_camera_frame is not None and len(self.current_camera_frame) > 0:
            try:
                # Decode and process camera frame
                if isinstance(self.current_camera_frame, str):
                    # If we received base64 string
                    image_data = base64.b64decode(self.current_camera_frame.split(',')[
                                                      1] if ',' in self.current_camera_frame else self.current_camera_frame)
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    img = self.current_camera_frame

                # Resize to game dimensions
                if img is not None:
                    img = cv2.resize(img, (self.GAME_WIDTH, self.GAME_HEIGHT))
                else:
                    # Fallback to stored background
                    img = self.game_background.copy()
            except Exception as e:
                print(f"Error processing camera frame: {e}")
                img = self.game_background.copy()
        else:
            # Fallback to stored background
            img = self.game_background.copy()

        # Apply semi-transparent overlay for better UI visibility
        overlay = np.zeros((self.GAME_HEIGHT, self.GAME_WIDTH, 3), dtype=np.uint8)
        overlay.fill(50)  # Dark overlay
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Render different elements based on game state
        if self.current_state == 'HOME':
            # Use existing Start-exit.png as background
            bg = self.backgrounds.get('home', self._create_gradient_background()).copy()

            # Add overlay text
            cv2.putText(bg, "ROCK PAPER SCISSORS", (self.GAME_WIDTH // 2 - 200, 60),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

            cv2.putText(bg, "Make a FIST to Start", (self.GAME_WIDTH // 2 - 150, self.GAME_HEIGHT - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display camera in a section of the screen
            if img is not None:
                img_resized = cv2.resize(img, (260, 260))
                bg[120:380, 520:780] = img_resized

            img = bg

        elif self.current_state == 'PREVIEW':
            # Preview screen - use existing background with game elements
            bg = self.backgrounds.get('game', self._create_gradient_background()).copy()

            cv2.putText(bg, "GAME PREVIEW", (self.GAME_WIDTH // 2 - 150, 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

            # Show the three gestures
            if 'rock' in self.resources and self.resources['rock'] is not None:
                rock_img = cv2.resize(self.resources['rock'], (100, 100))
                bg = self.overlay_image_with_alpha(bg, rock_img, 100, 200)
            else:
                cv2.putText(bg, "ROCK (fist)", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if 'paper' in self.resources and self.resources['paper'] is not None:
                paper_img = cv2.resize(self.resources['paper'], (100, 100))
                bg = self.overlay_image_with_alpha(bg, paper_img, 350, 200)
            else:
                cv2.putText(bg, "PAPER (open hand)", (310, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if 'scissors' in self.resources and self.resources['scissors'] is not None:
                scissors_img = cv2.resize(self.resources['scissors'], (100, 100))
                bg = self.overlay_image_with_alpha(bg, scissors_img, 600, 200)
            else:
                cv2.putText(bg, "SCISSORS (peace sign)", (550, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add gesture explanations
            cv2.putText(bg, "Fist = Rock", (80, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(bg, "Open Hand = Paper", (310, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(bg, "Victory Sign = Scissors", (550, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show countdown
            if hasattr(self, 'preview_timer'):
                cv2.putText(bg, f"Starting in: {int(self.preview_timer) + 1}", (self.GAME_WIDTH // 2 - 100, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display camera in a section of the screen
            if img is not None:
                img_resized = cv2.resize(img, (260, 260))
                bg[180:440, 570:830] = img_resized

            img = bg

        elif self.current_state == 'LEVEL_SELECT':
            # Level selection screen - use existing Yes-change.png background
            bg = self.backgrounds.get('level', self._create_gradient_background()).copy()

            cv2.putText(bg, "SELECT DIFFICULTY", (self.GAME_WIDTH // 2 - 180, 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

            # Display difficulty options
            cv2.putText(bg, "1 Finger = EASY (3 rounds)", (self.GAME_WIDTH // 2 - 200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(bg, "2 Fingers = MEDIUM (5 rounds)", (self.GAME_WIDTH // 2 - 200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(bg, "3 Fingers = HARD (7 rounds)", (self.GAME_WIDTH // 2 - 200, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display camera in a section of the screen
            if img is not None:
                img_resized = cv2.resize(img, (260, 260))
                bg[120:380, 500:760] = img_resized

            img = bg

        elif self.current_state == 'GAME_READY':
            # Ready screen - use existing Robot-player.png background
            bg = self.backgrounds.get('game', self._create_gradient_background()).copy()

            cv2.putText(bg, f"Round {self.current_round + 1} of {self.round_count}",
                        (self.GAME_WIDTH // 2 - 150, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

            # Display scores
            cv2.putText(bg, f"You: {self.score}", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(bg, f"Computer: {self.computer_score}", (self.GAME_WIDTH - 250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show countdown
            cv2.putText(bg, str(self.countdown), (self.GAME_WIDTH // 2 - 20, self.GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 255), 3)

            cv2.putText(bg, "Get Ready!", (self.GAME_WIDTH // 2 - 80, self.GAME_HEIGHT // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display camera in a section of the screen
            if img is not None:
                img_resized = cv2.resize(img, (260, 260))
                bg[180:440, 570:830] = img_resized

            img = bg

        elif self.current_state == 'GAME_MOVE':
            # Game screen - use existing Robot-player.png background
            bg = self.backgrounds.get('game', self._create_gradient_background()).copy()

            cv2.putText(bg, f"Round {self.current_round + 1} of {self.round_count}",
                        (self.GAME_WIDTH // 2 - 150, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

            # Display scores
            cv2.putText(bg, f"You: {self.score}", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(bg, f"Computer: {self.computer_score}", (self.GAME_WIDTH - 250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show time remaining
            time_left = max(0, round(self.move_timer, 1))
            cv2.putText(bg, f"Time: {time_left:.1f}s", (self.GAME_WIDTH // 2 - 80, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Show instructions
            cv2.putText(bg, "Make your move!", (self.GAME_WIDTH // 2 - 100, self.GAME_HEIGHT - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show player's current detected move if any
            if self.player_move:
                cv2.putText(bg, f"You chose: {self.player_move.upper()}",
                            (self.GAME_WIDTH // 2 - 120, self.GAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display camera in a section of the screen
            if img is not None:
                img_resized = cv2.resize(img, (260, 260))
                bg[180:440, 570:830] = img_resized

            img = bg

        elif self.current_state == 'ROUND_RESULT':
            # Round result screen - use game_score_bg.png background
            bg = self.backgrounds.get('result', self._create_gradient_background()).copy()

            cv2.putText(bg, f"Round {self.current_round} Result",
                        (self.GAME_WIDTH // 2 - 150, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

            # Show player and computer moves
            # Instead of overlaying images with cvzone, we'll just display text
            cv2.putText(bg, f"You: {self.player_move.upper() if self.player_move else 'NONE'}",
                        (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(bg, f"Computer: {self.computer_move.upper() if self.computer_move else 'NONE'}",
                        (500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show who won
            result_text = "DRAW!"
            result_color = (255, 255, 0)  # Yellow for draw

            if self.round_result == "player":
                result_text = "YOU WIN!"
                result_color = (0, 255, 0)  # Green for win
            elif self.round_result == "computer":
                result_text = "COMPUTER WINS!"
                result_color = (0, 0, 255)  # Red for loss

            cv2.putText(bg, result_text, (self.GAME_WIDTH // 2 - 120, self.GAME_HEIGHT // 2 + 100),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, result_color, 2)

            # Display scores
            cv2.putText(bg, f"You: {self.score}", (150, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(bg, f"Computer: {self.computer_score}", (500, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            img = bg

        elif self.current_state == 'GAME_OVER':
            # Game over screen - select appropriate background based on result
            if self.score > self.computer_score:
                bg = self.backgrounds.get('win', self._create_gradient_background()).copy()
                final_result = "YOU WIN!"
                result_color = (0, 255, 0)  # Green for win
            elif self.score < self.computer_score:
                bg = self.backgrounds.get('lose', self._create_gradient_background()).copy()
                final_result = "COMPUTER WINS!"
                result_color = (0, 0, 255)  # Red for loss
            else:
                bg = self.backgrounds.get('draw', self._create_gradient_background()).copy()
                final_result = "DRAW!"
                result_color = (255, 255, 0)  # Yellow for draw

            # Add additional text to the background
            cv2.putText(bg, "GAME OVER", (self.GAME_WIDTH // 2 - 150, 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2)

            cv2.putText(bg, final_result, (self.GAME_WIDTH // 2 - 120, self.GAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, result_color, 2)

            # Display final scores
            cv2.putText(bg, f"Final Score:", (self.GAME_WIDTH // 2 - 100, self.GAME_HEIGHT // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(bg, f"You: {self.score}  Computer: {self.computer_score}",
                        (self.GAME_WIDTH // 2 - 150, self.GAME_HEIGHT // 2 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show difficulty
            cv2.putText(bg, f"Difficulty: {self.difficulty}",
                        (self.GAME_WIDTH // 2 - 100, self.GAME_HEIGHT // 2 + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # No camera display needed on game over screen
            img = bg

        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"

        return None