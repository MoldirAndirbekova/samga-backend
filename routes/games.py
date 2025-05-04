from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Body, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from database import prisma
from routes.auth import get_current_user, SECRET_KEY, ALGORITHM
import numpy as np
import base64
import cv2
import json
import asyncio
from datetime import datetime
import uuid
import mediapipe as mp  # Add MediaPipe import
from .bubble_pop import BubblePopGameState  # Import BubblePopGameState
from .letter_tracing import LetterTracingGameState
from .fruit_slicer import FruitSlicerGameState
from .snake import SnakeGameState
from .constructor import ConstructorGameState  # Import ConstructorGameState
import jwt

router = APIRouter()

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe hands detector - initialize once for reuse
hands_detector = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video processing
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Game constants
GAME_WIDTH = 800
GAME_HEIGHT = 600
BALL_RADIUS = 15
PADDLE_WIDTH = 20
INITIAL_PADDLE_HEIGHT = 100
INITIAL_BALL_SPEED = 1
MAX_BALL_SPEED = 15
SPEED_INCREMENT = 0.2
HAND_MOVEMENT_MULTIPLIER = 1.5
DIFFICULTY_LEVELS = {
    "EASY": {"paddleHeight": 100, "speedMultiplier": 1},
    "MEDIUM": {"paddleHeight": 80, "speedMultiplier": 1.5},
    "HARD": {"paddleHeight": 60, "speedMultiplier": 2}
}

# Store active game sessions
active_games = {}

# Store game results for reporting
game_results = []


class GameCreate(BaseModel):
    name: str
    category_id: str


class GameResponse(BaseModel):
    id: str
    name: str
    category_id: str

    class Config:
        from_attributes = True


class HandTrackingRequest(BaseModel):
    image: str  # Base64 encoded image


class HandTrackingResponse(BaseModel):
    left: Optional[Dict[str, Any]] = None
    right: Optional[Dict[str, Any]] = None


class GameStartRequest(BaseModel):
    difficulty: str = "EASY"  # EASY, MEDIUM, HARD
    game_type: str = "ping_pong"  # ping_pong, bubble_pop, etc.
    child_id: Optional[str] = None  # Optional child ID to associate game with


class GameResultResponse(BaseModel):
    game_id: str
    difficulty: str
    score: int
    duration_seconds: int
    left_score: int
    right_score: int
    timestamp: str
    skills: Dict[str, float] = {}  # Added skills metrics
    child_id: Optional[str] = None  # Child ID if the game is played by a child


class GameReportResponse(BaseModel):
    total_games: int
    average_score: float
    average_duration: float
    games_by_difficulty: Dict[str, int]
    recent_games: List[GameResultResponse]
    skill_metrics: Dict[str, float] = {}  # Added overall skill metrics
    skill_progress: Dict[str, List[Dict[str, Any]]] = {}  # Added skill progress over time
    child_id: Optional[str] = None  # Child ID if the report is for a specific child


class GameState:
    def __init__(self, game_id, difficulty="EASY", child_id=None):
        self.game_id = game_id
        self.difficulty = difficulty
        self.child_id = child_id  # Track which child is playing the game
        self.paddle_height = DIFFICULTY_LEVELS[difficulty]["paddleHeight"]
        self.speed_multiplier = DIFFICULTY_LEVELS[difficulty]["speedMultiplier"]

        # Game state
        self.ball = {"x": GAME_WIDTH / 2, "y": GAME_HEIGHT / 2, "dx": INITIAL_BALL_SPEED * self.speed_multiplier,
                     "dy": INITIAL_BALL_SPEED * self.speed_multiplier}
        self.left_paddle = {"y": GAME_HEIGHT / 2 - self.paddle_height / 2}
        self.right_paddle = {"y": GAME_HEIGHT / 2 - self.paddle_height / 2}
        self.left_score = 0
        self.right_score = 0
        self.score = 0
        self.current_speed = INITIAL_BALL_SPEED * self.speed_multiplier
        self.game_over = False
        self.game_active = False
        self.start_time = None
        self.last_update = None

        # Hand tracking
        self.left_hand = None
        self.right_hand = None

        # FPS control
        self.fps = 60
        self.frame_time = 1 / self.fps

    def start_game(self):
        self.game_active = True
        self.game_over = False
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.left_score = 0
        self.right_score = 0
        self.score = 0
        self.ball = {"x": GAME_WIDTH / 2, "y": GAME_HEIGHT / 2, "dx": INITIAL_BALL_SPEED * self.speed_multiplier,
                     "dy": INITIAL_BALL_SPEED * self.speed_multiplier}
        self.current_speed = INITIAL_BALL_SPEED * self.speed_multiplier

    def update_hands(self, left_hand, right_hand):
        """Update paddle positions based on hand tracking data"""
        if left_hand:
            # Apply movement multiplier
            video_height = 480  # Standard camera height
            video_center = video_height / 2
            hand_offset = (left_hand["position"]["y"] - video_center) * HAND_MOVEMENT_MULTIPLIER
            game_y = GAME_HEIGHT / 2 + (hand_offset / video_height) * GAME_HEIGHT

            # Clamp to game boundaries
            game_y = min(max(game_y, self.paddle_height / 2), GAME_HEIGHT - self.paddle_height / 2)

            # Update paddle position
            self.left_paddle["y"] = game_y - self.paddle_height / 2
            self.left_hand = left_hand

        if right_hand:
            # Apply movement multiplier
            video_height = 480  # Standard camera height
            video_center = video_height / 2
            hand_offset = (right_hand["position"]["y"] - video_center) * HAND_MOVEMENT_MULTIPLIER
            game_y = GAME_HEIGHT / 2 + (hand_offset / video_height) * GAME_HEIGHT

            # Clamp to game boundaries
            game_y = min(max(game_y, self.paddle_height / 2), GAME_HEIGHT - self.paddle_height / 2)

            # Update paddle position
            self.right_paddle["y"] = game_y - self.paddle_height / 2
            self.right_hand = right_hand

    def update_game_state(self):
        """Update the game state for one frame"""
        if not self.game_active or self.game_over:
            return

        # Calculate delta time
        now = datetime.now()
        dt = (now - self.last_update).total_seconds()
        self.last_update = now

        # Move ball
        self.ball["x"] += self.ball["dx"] * dt * 200  # Adjust speed based on delta time
        self.ball["y"] += self.ball["dy"] * dt * 200

        # Ball collision with top and bottom walls
        if self.ball["y"] <= BALL_RADIUS or self.ball["y"] >= GAME_HEIGHT - BALL_RADIUS:
            self.ball["dy"] = -self.ball["dy"]

        # Ball collision with paddles
        # Left paddle
        if (
                self.ball["x"] - BALL_RADIUS <= PADDLE_WIDTH and
                self.ball["y"] >= self.left_paddle["y"] and
                self.ball["y"] <= self.left_paddle["y"] + self.paddle_height
        ):
            self.ball["dx"] = -self.ball["dx"]

            # Increase ball speed
            self.current_speed = min(MAX_BALL_SPEED, self.current_speed + SPEED_INCREMENT * self.speed_multiplier)
            self.ball["dx"] = self.current_speed if self.ball["dx"] > 0 else -self.current_speed

            # Update scores
            self.score += 1
            self.left_score += 1

        # Right paddle
        if (
                self.ball["x"] + BALL_RADIUS >= GAME_WIDTH - PADDLE_WIDTH and
                self.ball["y"] >= self.right_paddle["y"] and
                self.ball["y"] <= self.right_paddle["y"] + self.paddle_height
        ):
            self.ball["dx"] = -self.ball["dx"]

            # Increase ball speed
            self.current_speed = min(MAX_BALL_SPEED, self.current_speed + SPEED_INCREMENT * self.speed_multiplier)
            self.ball["dx"] = self.current_speed if self.ball["dx"] > 0 else -self.current_speed

            # Update scores
            self.score += 1
            self.right_score += 1

        # Game over if ball leaves game area on left or right
        if self.ball["x"] < 0 or self.ball["x"] > GAME_WIDTH:
            self.game_over = True
            self.game_active = False
            # Save game result when game is over
            self.save_game_result()

    def calculate_skill_metrics(self):
        """Calculate skill metrics based on game performance"""
        metrics = {}

        # Base metrics on game performance
        # Hand-eye coordination: Based on successful hits/total attempts
        total_attempts = self.left_score + self.right_score
        if total_attempts > 0:
            metrics["hand_eye_coordination"] = min(100, (self.score / total_attempts) * 100)
        else:
            metrics["hand_eye_coordination"] = 0

        # Agility: Based on speed of successful hits at higher speeds
        if self.current_speed > INITIAL_BALL_SPEED:
            agility_factor = (self.current_speed / MAX_BALL_SPEED) * 100
            metrics["agility"] = min(100, agility_factor)
        else:
            metrics["agility"] = 0

        # Focus: Based on consecutive successful hits without misses
        max_streak = max(self.left_score, self.right_score)
        focus_factor = (max_streak / 10) * 100  # 10 consecutive hits = 100% focus
        metrics["focus"] = min(100, focus_factor)

        # Reaction time: Inverse of speed (faster = better reaction time)
        if self.current_speed > INITIAL_BALL_SPEED:
            reaction_factor = (self.current_speed / MAX_BALL_SPEED) * 100
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
                "difficulty": self.difficulty,
                "score": self.score,
                "duration_seconds": int(duration),
                "left_score": self.left_score,
                "right_score": self.right_score,
                "timestamp": datetime.now().isoformat(),
                "skills": skill_metrics,
                "child_id": self.child_id  # Include child_id in game results
            }

            # Add to global results list
            game_results.append(result)

            # Keep only the last 100 results to avoid memory issues
            if len(game_results) > 100:
                game_results.pop(0)

            # Save to database using Prisma
            asyncio.create_task(self._persist_to_database(result, skill_metrics))

    async def _persist_to_database(self, result, skill_metrics):
        """Persist game result to database using Prisma"""
        try:
            # Check if the database is connected
            if not prisma.is_connected():
                await prisma.connect()

            # Use ping-pong game type ID
            game_type_id = "ping-pong"

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

            print(f"Saved ping pong game report to database with ID: {game_report.id}")

        except Exception as e:
            print(f"Error saving ping pong game report to database: {str(e)}")

    def render_frame(self):
        """Render the current game state to an image"""
        print(f"Rendering PingPong frame: active={self.game_active}, score={self.score}")

        # Create a blank canvas
        img = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)

        # Draw background - dark green
        cv2.rectangle(img, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 100, 0), -1)

        # Draw center line
        cv2.line(img, (GAME_WIDTH // 2, 0), (GAME_WIDTH // 2, GAME_HEIGHT), (255, 255, 255), 2)
        cv2.circle(img, (GAME_WIDTH // 2, GAME_HEIGHT // 2), 50, (255, 255, 255), 2)

        # Draw ball
        cv2.circle(
            img,
            (int(self.ball["x"]), int(self.ball["y"])),
            BALL_RADIUS,
            (255, 255, 255),  # White
            -1
        )

        # Draw paddles
        # Left paddle
        cv2.rectangle(
            img,
            (0, int(self.left_paddle["y"])),
            (PADDLE_WIDTH, int(self.left_paddle["y"] + self.paddle_height)),
            (0, 0, 255),  # Red
            -1
        )

        # Right paddle
        cv2.rectangle(
            img,
            (GAME_WIDTH - PADDLE_WIDTH, int(self.right_paddle["y"])),
            (GAME_WIDTH, int(self.right_paddle["y"] + self.paddle_height)),
            (0, 255, 0),  # Green
            -1
        )

        # Draw score
        cv2.putText(
            img,
            str(self.left_score),
            (GAME_WIDTH // 4, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            str(self.right_score),
            (3 * GAME_WIDTH // 4, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Draw difficulty
        cv2.putText(
            img,
            f"Difficulty: {self.difficulty}",
            (GAME_WIDTH // 2 - 80, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Draw speed
        cv2.putText(
            img,
            f"Speed: {int(self.current_speed)}",
            (GAME_WIDTH // 2 - 60, GAME_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Game over screen
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            cv2.putText(
                img,
                "GAME OVER",
                (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2 - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3
            )

            cv2.putText(
                img,
                f"Final Score: {self.score}",
                (GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Left Score: {self.left_score} | Right Score: {self.right_score}",
                (GAME_WIDTH // 2 - 200, GAME_HEIGHT // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Difficulty: {self.difficulty}",
                (GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Convert to base64 for sending over WebSocket
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"

        return None


# Load Haar cascade for hand detection (will fall back to this if other methods fail)
try:
    # Try to load pre-trained hand cascade
    hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'aGest.xml')
except:
    hand_cascade = None
    print("Warning: Hand cascade model could not be loaded")


# Function to calculate skill metrics for Bubble Pop game
def calculate_bubble_pop_skill_metrics(game):
    """Calculate skill metrics for Bubble Pop game"""
    metrics = {}

    # Hand-eye coordination: Based on successful pops vs penalties
    total_attempts = game.score + game.penalties
    if total_attempts > 0:
        metrics["hand_eye_coordination"] = min(100, (game.score / total_attempts) * 100)
    else:
        metrics["hand_eye_coordination"] = 0

    # Agility: Based on score within time limit
    agility_factor = (game.score / 50) * 100  # 50 pops in 60 sec = 100% agility
    metrics["agility"] = min(100, agility_factor)

    # Focus: Inverse of penalties (fewer penalties = better focus)
    if game.penalties == 0 and game.score > 0:
        metrics["focus"] = 100
    elif total_attempts > 0:
        focus_factor = 100 - ((game.penalties / total_attempts) * 100)
        metrics["focus"] = max(0, focus_factor)
    else:
        metrics["focus"] = 0

    # Reaction time: Based on score within time limit
    reaction_factor = (game.score / 40) * 100  # 40 pops in 60 sec = 100% reaction
    metrics["reaction_time"] = min(100, reaction_factor)

    return metrics


# Override BubblePopGameState save_game_result method
def save_bubble_pop_game_result(self):
    """Save the game result for Bubble Pop with skill metrics"""
    if self.start_time:
        duration = (datetime.now() - self.start_time).total_seconds()

        # Calculate skill metrics
        skill_metrics = calculate_bubble_pop_skill_metrics(self)

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
            "child_id": self.child_id  # Include child_id in game results
        }

        # Add to global results list
        game_results.append(result)
        print(f"Saved game result with skills: {skill_metrics}")

        # Keep only the last 100 results to avoid memory issues
        if len(game_results) > 100:
            game_results.pop(0)

        # Save to database using Prisma
        asyncio.create_task(self._persist_to_database(result, skill_metrics))


async def _persist_to_database(self, result, skill_metrics):
    """Persist game result to database using Prisma"""
    try:
        # Check if the database is connected
        if not prisma.is_connected():
            await prisma.connect()

        # Use bubble-pop game type ID
        game_type_id = "bubble-pop"

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

        print(f"Saved bubble pop game report to database with ID: {game_report.id}")

    except Exception as e:
        print(f"Error saving bubble pop game report to database: {str(e)}")


# Apply the override and persistent method to BubblePopGameState
BubblePopGameState.save_game_result = save_bubble_pop_game_result
BubblePopGameState._persist_to_database = _persist_to_database


@router.post("/game/start", response_model=Dict[str, Any])
async def create_game(request: GameStartRequest, current_user=Depends(get_current_user)):
    """Create a new game session and return the game ID"""
    game_id = str(uuid.uuid4())

    print(f"Creating game with type: {request.game_type}")

    if request.game_type == "bubble_pop":
        print(f"Initializing Bubble Pop game with difficulty: {request.difficulty}")
        active_games[game_id] = BubblePopGameState(game_id, request.difficulty, request.child_id)
    elif request.game_type == "letter_tracing":
        print(f"Initializing Letter Tracing game with difficulty: {request.difficulty}")
        active_games[game_id] = LetterTracingGameState(game_id, request.difficulty, request.child_id)
    elif request.game_type == "fruit_slicer":
        print(f"Initializing Fruit Slicer game with difficulty: {request.difficulty}")
        active_games[game_id] = FruitSlicerGameState(game_id, request.difficulty, request.child_id)
    elif request.game_type == "snake":
        print(f"Initializing Snake game with difficulty: {request.difficulty}")
        active_games[game_id] = SnakeGameState(game_id, request.difficulty, request.child_id)
    elif request.game_type == "constructor":
        print(f"Initializing Constructor game with difficulty: {request.difficulty}")
        active_games[game_id] = ConstructorGameState(game_id, request.difficulty, request.child_id)
    else:
        # Default to PingPong game
        print(f"Initializing PingPong game with difficulty: {request.difficulty}")
        active_games[game_id] = GameState(game_id, request.difficulty, request.child_id)

    return {
        "game_id": game_id,
        "message": "Game created successfully",
        "difficulty": request.difficulty,
        "game_type": request.game_type,
        "child_id": request.child_id
    }
@router.websocket("/game/{game_id}/ws")
async def game_websocket(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for game rendering and control"""
    # Validate the token from query parameters
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return

    # Verify the token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            await websocket.close(code=1008, reason="Invalid token")
            return

        # We could load the user here if needed
        # user = await prisma.user.find_unique(where={"email": email})
        # if user is None:
        #    await websocket.close(code=1008, reason="User not found")
        #    return
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid authentication")
        return

    await websocket.accept()

    if game_id not in active_games:
        await websocket.send_json({"error": "Game not found"})
        await websocket.close()
        return

    game = active_games[game_id]

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "start_game":
                game.start_game()

            elif message["type"] == "hand_tracking_image":
                # Process the image for hand tracking AND update camera frame for AR
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(message["data"]["image"].split(',')[1] if ',' in message["data"]["image"] else message["data"]["image"])
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        if isinstance(game, FruitSlicerGameState):
                            # Process the image for pose tracking
                            pose_result = process_image_for_pose(img)

                            # Update nose position in game state
                            if pose_result and "nose" in pose_result:
                                game.update_pose(pose_result["nose"])

                                # Send tracking results back to client
                                await websocket.send_json({
                                    "type": "pose_tracking_result",
                                    "data": pose_result
                                })
                        else:
                            # Process the image for hand tracking
                            result = process_image_for_hands(img)

                            # Update hands/hand in game state
                            if hasattr(game, 'update_hands'):
                                # For games that use multiple hands
                                left_hand = result.left
                                right_hand = result.right
                                game.update_hands(left_hand, right_hand)
                            elif hasattr(game, 'update_hand'):
                                # For games that use single hand (like letter tracing)
                                # Use the first detected hand (left or right)
                                hand = result.left if result.left else result.right
                                if hand:
                                    game.update_hand(hand)

                        # Update the camera frame for AR overlay
                        if hasattr(game, 'update_camera_frame'):
                            game.update_camera_frame(img)

                        # Send tracking results back to client
                        await websocket.send_json({
                            "type": "hand_tracking_result",
                            "data": {
                                "left": result.left,
                                "right": result.right
                            }
                        })
                except Exception as e:
                    print(f"Error processing hand tracking image: {str(e)}")

            elif message["type"] == "update_score" and hasattr(game, 'score'):
                # For games like Bubble Pop that need explicit score updates
                if "score_increment" in message["data"]:
                    game.score += message["data"]["score_increment"]

            elif message["type"] == "close_game":
                # Clean disconnect requested by client
                print(f"Client requested clean disconnect for game {game_id}")
                if game_id in active_games:
                    # Save game result if game is active
                    if game.game_active and not game.game_over:
                        game.game_over = True
                        game.game_active = False
                        try:
                            game.save_game_result()
                        except Exception as e:
                            print(f"Error saving game result on clean disconnect: {str(e)}")

                    # Remove from active games
                    del active_games[game_id]

                # Send acknowledgment
                await websocket.send_json({
                    "type": "close_acknowledgment",
                    "data": {"status": "success"}
                })

                # Close the connection
                await websocket.close()
                return

            # Update game state
            game.update_game_state()

            # Render frame and send to client
            frame = game.render_frame()

            # Prepare game state data to send
            # In the WebSocket handler, modify the game_state_data preparation:
            game_state_data = {
                "frame": frame,
                "game_active": game.game_active,
                "game_over": game.game_over,
            }

            # Add score and time info, handling different game types
            if hasattr(game, 'score'):
                game_state_data["score"] = game.score

            if hasattr(game, 'time_remaining'):
                game_state_data["time_remaining"] = game.time_remaining

            # Add combo information for Fruit Slicer
            if hasattr(game, 'combo'):
                game_state_data["combo"] = game.combo
                game_state_data["max_combo"] = game.max_combo

            # Add fruits data for Fruit Slicer
            if hasattr(game, 'fruits'):
                # Send basic fruit data (position, size, state)
                game_state_data["fruits"] = [
                    {
                        "id": fruit.id,
                        "x": fruit.x,
                        "y": fruit.y,
                        "size": fruit.size,
                        "sliced": fruit.sliced,
                        "is_bomb": getattr(fruit, 'is_bomb', False)
                    }
                    for fruit in game.fruits
                ]

            # Send game state
            await websocket.send_json({
                "type": "game_state",
                "data": game_state_data
            })

            await asyncio.sleep(game.frame_time)

    except WebSocketDisconnect:
        print(f"Client disconnected from game {game_id}")
        if game_id in active_games:
            # Save game result if game was active
            if game.game_active and not game.game_over:
                game.game_over = True
                game.game_active = False
                try:
                    game.save_game_result()
                except Exception as e:
                    print(f"Error saving game result on disconnect: {str(e)}")

            del active_games[game_id]
    except Exception as e:
        print(f"Error in game WebSocket: {str(e)}")
        if game_id in active_games:
            del active_games[game_id]


def process_image_for_pose(img):
    """
    Process an image to detect face/pose using MediaPipe Face Mesh.
    Returns nose position for Fruit Slicer game.
    """
    height, width = img.shape[:2]
    result = {}

    # Initialize MediaPipe Face Mesh solution
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Convert image to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with Face Mesh
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]

        # Get nose tip (landmark index 4)
        nose_landmark = face_landmarks.landmark[4]

        # Convert normalized coordinates to pixel coordinates
        nose_x = nose_landmark.x * width
        nose_y = nose_landmark.y * height

        # Add nose position to result
        result["nose"] = {
            "x": float(nose_x),
            "y": float(nose_y),
            "score": 1.0
        }

    # If Face Mesh fails, fallback to MediaPipe Pose (like in original code)
    if "nose" not in result:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        pose_results = pose.process(rgb_image)

        if pose_results.pose_landmarks:
            # Get nose landmark
            nose_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Convert to pixel coordinates
            nose_x = nose_landmark.x * width
            nose_y = nose_landmark.y * height

            # Add to result
            result["nose"] = {
                "x": float(nose_x),
                "y": float(nose_y),
                "score": float(nose_landmark.visibility)
            }

    return result

@router.post("/handtracking", response_model=HandTrackingResponse)
async def track_hands(request: HandTrackingRequest):
    """
    Process an image and return hand tracking data.
    The image should be base64 encoded.
    """
    try:
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        result = process_image_for_hands(img)

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

def process_image_for_hands(img):
    """
    Process an image to detect hands using MediaPipe Hands solution.
    Returns hand landmarks processed into a format usable by the game.
    """
    height, width = img.shape[:2]
    result = HandTrackingResponse()

    left_hand = None
    right_hand = None

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_results = hands_detector.process(rgb_image)

    if mp_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({
                    "x": lm.x * width,
                    "y": lm.y * height,
                    "z": lm.z * width
                })

            # Calculate both palm center and index finger tip

            # Palm center calculation (for backward compatibility with existing games)
            palm_landmarks = [
                landmarks[0],  # Wrist
                landmarks[5],  # Index finger base
                landmarks[9],  # Middle finger base
                landmarks[13], # Ring finger base
                landmarks[17]  # Pinky finger base
            ]

            palm_center_x = sum(lm["x"] for lm in palm_landmarks) / len(palm_landmarks)
            palm_center_y = sum(lm["y"] for lm in palm_landmarks) / len(palm_landmarks)

            # Index finger tip (landmark 8)
            index_finger_tip = landmarks[8]

            # Calculate bounding box
            x_coords = [lm["x"] for lm in landmarks]
            y_coords = [lm["y"] for lm in landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Determine hand type
            hand_type = "unknown"
            if mp_results.multi_handedness and len(mp_results.multi_handedness) > hand_idx:
                handedness = mp_results.multi_handedness[hand_idx].classification[0].label
                hand_type = handedness.lower()
            else:
                # Fallback to position-based determination
                hand_type = "left" if palm_center_x < width / 2 else "right"

            area = bbox_width * bbox_height

            # Create hand object with both palm center and index finger tip
            hand = {
                "position": {"x": float(palm_center_x), "y": float(palm_center_y)},  # Maintain backward compatibility
                "palm_center": {"x": float(palm_center_x), "y": float(palm_center_y)},
                "index_finger_tip": {"x": float(index_finger_tip["x"]), "y": float(index_finger_tip["y"])},
                "bbox": {
                    "x": float(x_min),
                    "y": float(y_min),
                    "width": float(bbox_width),
                    "height": float(bbox_height)
                },
                "score": 1.0,
                "area": float(area),
                "landmarks": landmarks,
                "handedness": hand_type
            }

            if hand_type == "left" and (left_hand is None or hand["area"] > left_hand["area"]):
                left_hand = hand
            elif hand_type == "right" and (right_hand is None or hand["area"] > right_hand["area"]):
                right_hand = hand

    if left_hand is not None:
        result.left = left_hand

    if right_hand is not None:
        result.right = right_hand

    return result

@router.get("/categories", response_model=List[dict])
async def get_categories(current_user = Depends(get_current_user)):
    categories = await prisma.category.find_many()
    return [{"id": cat.id, "name": cat.name} for cat in categories]

@router.get("/", response_model=List[GameResponse])
async def get_games(current_user = Depends(get_current_user)):
    games = await prisma.game.find_many()
    return [
        GameResponse(
            id=game.id,
            name=game.name,
            category_id=game.category_id
        )
        for game in games
    ]

@router.get("/{game_id}", response_model=GameResponse)
async def get_game(game_id: str, current_user = Depends(get_current_user)):
    game = await prisma.game.find_unique(
        where={"id": game_id}
    )
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    return GameResponse(
        id=game.id,
        name=game.name,
        category_id=game.category_id
    )

@router.get("/category/{category_id}", response_model=List[GameResponse])
async def get_games_by_category(category_id: str, current_user = Depends(get_current_user)):
    games = await prisma.game.find_many(
        where={"category_id": category_id}
    )
    return [
        GameResponse(
            id=game.id,
            name=game.name,
            category_id=game.category_id
        )
        for game in games
    ]

@router.get("/results", response_model=List[GameResultResponse])
async def get_game_results(current_user = Depends(get_current_user)):
    """Return all stored game results"""
    return [GameResultResponse(**result) for result in game_results]

class SaveGameReportRequest(BaseModel):
    """Request model for saving a game report via REST API"""
    game_id: str
    game_type: str  # "ping_pong" or "bubble_pop"
    difficulty: str
    score: int
    duration_seconds: int
    left_score: int  # For ping pong: left paddle score, for bubble pop: total score
    right_score: int  # For ping pong: right paddle score, for bubble pop: penalties
    skills: Dict[str, float]
    child_id: Optional[str] = None

@router.post("/results/save", response_model=Dict[str, Any])
async def save_game_report(request: SaveGameReportRequest, current_user = Depends(get_current_user)):
    """Save a game report via REST API"""
    try:
        # Check if the database is connected
        if not prisma.is_connected():
            await prisma.connect()

        # Determine game type ID
        game_type_id = "ping-pong"
        if request.game_type == "bubble_pop":
            game_type_id = "bubble-pop"

        # Create the game report in the database
        game_report = await prisma.gamereport.create(
            data={
                "gameId": request.game_id,
                "gameTypeId": game_type_id,
                "childId": request.child_id,
                "difficulty": request.difficulty,
                "score": request.score,
                "leftScore": request.left_score,
                "rightScore": request.right_score,
                "durationSeconds": request.duration_seconds,
                "skillMetrics": {
                    "create": [
                        {"skillName": skill, "value": value}
                        for skill, value in request.skills.items()
                    ]
                }
            },
            include={"skillMetrics": True}
        )

        # Also add to in-memory results for backward compatibility
        result = {
            "game_id": request.game_id,
            "difficulty": request.difficulty,
            "score": request.score,
            "duration_seconds": request.duration_seconds,
            "left_score": request.left_score,
            "right_score": request.right_score,
            "timestamp": datetime.now().isoformat(),
            "skills": request.skills,
            "child_id": request.child_id
        }

        game_results.append(result)

        # Keep only the last 100 results in memory
        if len(game_results) > 100:
            game_results.pop(0)

        return {
            "success": True,
            "message": "Game report saved successfully",
            "report_id": game_report.id
        }

    except Exception as e:
        print(f"Error saving game report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save game report: {str(e)}"
        )

@router.get("/results/report", response_model=GameReportResponse)
async def get_game_report(child_id: Optional[str] = None, current_user = Depends(get_current_user)):
    """Generate a summary report of game results with skill metrics"""
    # First try to get results from database
    try:
        # Check if the database is connected
        if not prisma.is_connected():
            await prisma.connect()

        # Build where condition based on child_id
        where = {}
        if child_id:
            where["childId"] = child_id

        # Fetch game reports from database
        db_game_reports = await prisma.gamereport.find_many(
            where=where,
            include={"skillMetrics": True},
            order_by={"timestamp": "desc"}
        )

        # If reports exist in the database, use them
        if db_game_reports:
            print(f"Found {len(db_game_reports)} game reports in database")

            # Convert database results to game report format
            filtered_results = []
            for report in db_game_reports:
                # Map skill metrics to dictionary
                skills = {}
                for metric in report.skillMetrics:
                    skills[metric.skillName] = metric.value

                # Create game result
                filtered_results.append({
                    "game_id": report.gameId,
                    "difficulty": report.difficulty,
                    "score": report.score,
                    "duration_seconds": report.durationSeconds,
                    "left_score": report.leftScore,
                    "right_score": report.rightScore,
                    "timestamp": report.timestamp.isoformat(),
                    "skills": skills,
                    "child_id": report.childId
                })
        else:
            # Fall back to in-memory results if database has no data
            print("No game reports found in database, using in-memory data")
            filtered_results = [r for r in game_results if child_id is None or r.get("child_id") == child_id]
    except Exception as e:
        print(f"Error fetching from database, using in-memory data: {str(e)}")
        # Fall back to in-memory results
        filtered_results = [r for r in game_results if child_id is None or r.get("child_id") == child_id]

    if not filtered_results:
        return GameReportResponse(
            total_games=0,
            average_score=0,
            average_duration=0,
            games_by_difficulty={},
            recent_games=[],
            skill_metrics={},
            skill_progress={},
            child_id=child_id
        )

    total_games = len(filtered_results)
    avg_score = sum(r["score"] for r in filtered_results) / total_games if total_games > 0 else 0
    avg_duration = sum(r["duration_seconds"] for r in filtered_results) / total_games if total_games > 0 else 0

    difficulty_counts = {}
    for result in filtered_results:
        difficulty = result["difficulty"]
        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1
        else:
            difficulty_counts[difficulty] = 1

    # Calculate overall skill metrics from all games
    skill_metrics = {
        "hand_eye_coordination": 0,
        "agility": 0,
        "focus": 0,
        "reaction_time": 0
    }

    # Track skill progress over time (last 10 games for each skill)
    skill_progress = {
        "hand_eye_coordination": [],
        "agility": [],
        "focus": [],
        "reaction_time": []
    }

    skill_count = 0
    for result in filtered_results:
        if "skills" in result and result["skills"]:
            for skill, value in result["skills"].items():
                if skill in skill_metrics:
                    skill_metrics[skill] += value

            skill_count += 1

            # Add to progress tracking with timestamp (last 10 games)
            game_time = datetime.fromisoformat(result["timestamp"])
            for skill in skill_progress.keys():
                if skill in result.get("skills", {}):
                    skill_progress[skill].append({
                        "timestamp": game_time.strftime("%Y-%m-%d %H:%M"),
                        "value": result["skills"][skill],
                        "game_type": "bubble_pop" if result.get("right_score", 0) == result.get("penalties", 0) else "ping_pong",
                        "difficulty": result["difficulty"]
                    })

    # Average skill metrics
    if skill_count > 0:
        for skill in skill_metrics:
            skill_metrics[skill] = round(skill_metrics[skill] / skill_count, 2)

    # Keep only the last 10 records for each skill
    for skill in skill_progress:
        skill_progress[skill] = sorted(skill_progress[skill], key=lambda x: x["timestamp"])[-10:]

    recent_games = [GameResultResponse(**r) for r in filtered_results[-10:]]

    return GameReportResponse(
        total_games=total_games,
        average_score=round(avg_score, 2),
        average_duration=round(avg_duration, 2),
        games_by_difficulty=difficulty_counts,
        recent_games=recent_games,
        skill_metrics=skill_metrics,
        skill_progress=skill_progress,
        child_id=child_id
    )