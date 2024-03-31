# ~ BarakXYZ - XYZ Emoji - 2024 - CS50P Final Project ~

import cv2
import mediapipe as mp
import time
import sys
import pygame
import numpy as np
from cv_helpers import calculate_fps
from emojis_map import emojis
import sounddevice as sd
import soundfile as sf
import threading
import time
import os


class VideoCapture:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        threading.Thread(target=self.update, args=()).start()

    def update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.is_running = False
        self.cap.release()


def play_sound(file_path, volume=1.0):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Check if volume is within a reasonable range
    if volume < 0.0 or volume > 1.0:
        print("Error: Volume must be between 0.0 and 1.0.")
        return

    try:
        # Load the file
        data, fs = sf.read(file_path)

        # Adjust volume
        data = data * volume

        # Play sound
        sd.play(data, fs)
        sd.wait()  # Wait until the sound has finished playing

    except Exception as e:
        print(f"Couldn't load the sound file: {e}")


def release_exit(cap, goodbye_message="Cya_Bobik", exit_code="0"):
    print(goodbye_message)
    try:  # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Couldn't close and release the camera capture: {e}")
        sys.exit(1)


def calculate_distance(hand_lmks, mp_hands):
    thumb_tip = hand_lmks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_lmks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )
    return distance


# Initialize Pygame's mixer module
pygame.mixer.init(
    frequency=44100, size=-16, channels=2, buffer=64
)  # Adjust buffer size as needed
pygame.mixer.set_num_channels(64)  # Increase the number of available channels

# Load your sound file
snare_01 = pygame.mixer.Sound("Sounds/snare_01.wav")
kick_01 = pygame.mixer.Sound("Sounds/kick_01.wav")


# Function to play sound
def play_sound_pygame(snare=False, kick=False):
    if snare:
        snare_01.play()
        return
    if kick:
        kick_01.play()
        return


# Overlay the emojis on the corresponding landmark (hand tracking points)
def overlay_emoji(img, img_dim, position, emoji):
    img_h, img_w, img_c = img_dim
    lmrk_x, lmrk_y = position  # Unpack the position tuple
    eh, ew, ec = (
        emoji.shape
    )  # Get emoji dimensions; emoji height (eh), emoji width (ew), emoji channels (ec)
    if (
        0 <= lmrk_x - ew // 2 < img_w - ew and 0 <= lmrk_y - eh // 2 < img_h - eh
    ):  # Check if the emoji is within the frame
        alpha_s = emoji[:, :, 3] / 255.0  # Extract the alpha channel
        alpha_l = 1.0 - alpha_s  # Invert the alpha channel

        for c in range(0, 3):  # Loop through the RGB channels
            img[
                lmrk_y - eh // 2 : lmrk_y + eh // 2,
                lmrk_x - ew // 2 : lmrk_x + ew // 2,
                c,
            ] = (
                alpha_s * emoji[:, :, c]
                + alpha_l
                * img[
                    lmrk_y - eh // 2 : lmrk_y + eh // 2,
                    lmrk_x - ew // 2 : lmrk_x + ew // 2,
                    c,
                ]
            )


# Detecting the pose of the hand based on the finger positions
def check_pose(
    thumb_folded,
    index_folded,
    middle_folded,
    ring_folded,
    pinky_folded,
    debug=False,
    pose="open_hand",
):
    if (
        not thumb_folded
        and not index_folded
        and not middle_folded
        and not ring_folded
        and not pinky_folded
    ):
        pose = "open_hand"
    elif (
        thumb_folded
        and not index_folded
        and middle_folded
        and ring_folded
        and not pinky_folded
    ):
        pose = "rock_on"
    elif (
        not thumb_folded
        and not index_folded
        and middle_folded
        and ring_folded
        and not pinky_folded
    ):
        pose = "i_love_you"
    elif (
        thumb_folded
        and not index_folded
        and not middle_folded
        and ring_folded
        and pinky_folded
    ):
        pose = "peace"
    elif (
        not thumb_folded
        and index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    ):
        pose = "thumb_up"
    elif (
        thumb_folded and index_folded and middle_folded and ring_folded and pinky_folded
    ):
        pose = "fist"

    if debug:
        print(f"Pose: {pose}")
    return pose


# Detecting if a finger is folded or unfolded (True if folded, False if unfolded)
def is_finger_folded(
    handLms, img_w, img_h, tip_id, pip_id, mcp_id=None, is_thumb=False
):
    if is_thumb:
        pip_x = handLms.landmark[pip_id].x * img_w
        mcp_x = handLms.landmark[mcp_id].x * img_w
        return pip_x < mcp_x

    else:
        tip_y = handLms.landmark[tip_id].y * img_h
        pip_y = handLms.landmark[pip_id].y * img_h
        return tip_y > pip_y


def create_screen_text(
    img, text, position, font_scale=1, color=(255, 255, 255), thickness=2
):
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_PLAIN,
        font_scale,
        color,
        thickness,
    )


def main():
    # print(cv2.getBuildInformation())  # Debug
    # Initialize distances and sound trigger flags for both hands
    distance_right_hand = 2
    distance_left_hand = 2
    can_trigger_sound_right = True
    can_trigger_sound_left = True

    distance = 5
    distance_threshold = 0.03  # Distance threshold to trigger sound
    interval_time = 2  # Time interval to calculate BPM

    click_times = []  # List to store the times of the clicks
    intervals = []  # List to store the intervals between clicks
    num_intervals_for_average = 4  # Number of intervals to average over
    bpm = 0

    cap = VideoCapture(0)
    if not cap.cap.isOpened():
        print("Error: Could not open video device.")
    else:
        print("Opened video device")
        # cap = cv2.VideoCapture(4)  # Video input
        #   cap.cap.set(cv2.CAP_PROP_FPS, 60.0)
        #     # Set resolution to 640x480
        #     cap.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #     cap.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Get the camera frame dimensions
        img_w = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get the camera fps
        fps = cap.cap.get(cv2.CAP_PROP_FPS)

    mp_hands = mp.solutions.hands  # Init hand module
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # mp_face_detection = mp.solutions.face_detection
    # face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    mp_draw = (
        mp.solutions.drawing_utils
    )  # Initializing the drawing module, to draw landmarks on the hands

    previous_time = time.time()  # Initializing the previous time to calculate FPS

    while True:
        success, img = cap.read()
        if not success:
            print("Could not read camera input")
            break

        # Flip the frame horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        img_h, img_w, img_c = img.shape  # Get camera frame dimensions
        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # media pipe accepts RGB images only
        results_hands = hands.process(
            img_rgb
        )  # Process the converted image to detect hands (landmarks)

        # results_face = face_detection.process(
        #     img_rgb
        # )  # Process the converted image to detect faces

        # Checking if hands detected:
        if results_hands.multi_hand_landmarks:
            for hand_lmks, handedness in zip(
                results_hands.multi_hand_landmarks, results_hands.multi_handedness
            ):
                # Determine the hand type
                hand_type = handedness.classification[0].label

                # Calculate the distance for the current hand
                distance = calculate_distance(hand_lmks, mp_hands)

                # Process the right hand
                if hand_type == "Right":
                    distance_right_hand = distance
                    if (
                        distance_right_hand < distance_threshold
                        and can_trigger_sound_right
                    ):
                        play_sound_pygame(
                            kick=True
                        )  # Adjust the sound for the right hand
                        can_trigger_sound_right = False

                        # Calculate BPM
                        click_times.append(time.time())  # Record time of the click
                        if len(click_times) > 1:
                            # Calculate interval between the last two clicks
                            interval = click_times[-1] - click_times[-2]
                            intervals.append(interval)

                            # Keep only the last few intervals
                            if len(intervals) > num_intervals_for_average:
                                intervals.pop(0)

                            # Calculate the average interval if we have enough intervals
                            if len(intervals) == num_intervals_for_average:
                                average_interval = sum(intervals) / len(intervals)
                                # Convert the average interval to BPM
                                bpm = 60 / average_interval
                                print(
                                    f"Average BPM (last {num_intervals_for_average} intervals): {bpm:.2f}"
                                )

                if distance_right_hand > distance_threshold:
                    can_trigger_sound_right = True

                # Process the left hand
                if hand_type == "Left":
                    distance_left_hand = distance
                    if (
                        distance_left_hand < distance_threshold
                        and can_trigger_sound_left
                    ):
                        play_sound_pygame(
                            snare=True
                        )  # Adjust the sound for the left hand
                        can_trigger_sound_left = False

                    if distance_left_hand > distance_threshold:
                        can_trigger_sound_left = True

                # ---------------------------------------------- #

                # thumb_folded = is_finger_folded(handLms, img_w, img_h, 4, 1)
                thumb_folded = is_finger_folded(
                    hand_lmks, img_w, img_h, 4, 3, 2, is_thumb=True
                )
                index_folded = is_finger_folded(hand_lmks, img_w, img_h, 8, 6)
                middle_folded = is_finger_folded(hand_lmks, img_w, img_h, 12, 10)
                ring_folded = is_finger_folded(hand_lmks, img_w, img_h, 16, 14)
                pinky_folded = is_finger_folded(hand_lmks, img_w, img_h, 20, 18)

                pose = check_pose(
                    thumb_folded,
                    index_folded,
                    middle_folded,
                    ring_folded,
                    pinky_folded,
                )

                for id, lm in enumerate(hand_lmks.landmark):
                    nrml_lm_x, nrml_lm_y = int(lm.x * img_w), int(
                        lm.y * img_h
                    )  # Get the x and y coordinates of the landmark in the frame (in pixels)

                    if id == 4:
                        # print(nrml_lm_x, nrml_lm_y)
                        # print(lm.x, lm.y)
                        ...

                    if pose not in emojis:
                        continue

                    for emoji_info in emojis[pose]:
                        if id in emoji_info["landmarks"]:
                            overlay_emoji(
                                img,
                                (img_h, img_w, img_c),
                                (nrml_lm_x, nrml_lm_y),
                                emoji_info["emoji"],
                            )

                # ---------------------------------------------- #
                # Debugging
                # mp_draw.draw_landmarks(
                #     img, hand_lmks, mp_hands.HAND_CONNECTIONS
                # )  # Draw the landmarks on the hands

                # ---------------------------------------------- #

        # Draw the FPS on the frame
        create_screen_text(img, f"FPS: {int(fps)}", (10, 30))

        # Debug left hand distance from triggering sound
        create_screen_text(
            img, f"Left hand distance: {distance_left_hand:.2f}", (10, 70)
        )

        # Debug right hand distance from triggering sound
        create_screen_text(
            img, f"Right hand distance: {distance_right_hand:.2f}", (10, 90)
        )

        # Debug distance threshold
        create_screen_text(
            img, f"Distance threshold: {distance_threshold:.2f}", (10, 110)
        )

        # Debug FPS
        create_screen_text(img, f"FPS: {int(fps)}", (10, 150))

        # Debug camera dimensions
        create_screen_text(img, f"Camera dimensions: {img_w}x{img_h}", (10, 190))

        # Debug BPM
        create_screen_text(img, f"BPM: {bpm:.2f}", (10, 230))

        # Increase distance on pressing k
        if cv2.waitKey(1) == 107:
            distance_threshold += 0.01
        # Decrease distance on pressing j
        elif cv2.waitKey(1) == 106:
            if distance_threshold > 0.01:
                distance_threshold -= 0.01
        elif cv2.waitKey(1) == 27:
            break  # Exit when the escape key is pressed

        # previous_time = calculate_fps(previous_time, img)  # Calculate FPS
        cv2.imshow("Image", img)

    release_exit(cap)


if __name__ == "__main__":
    main()
