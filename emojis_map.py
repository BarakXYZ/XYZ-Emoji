# ~ BarakXYZ - XYZ Emoji - 2024 - CS50P Final Project ~

import cv2
import os

# Load emojis
emoji_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")

emojis = {
    "tip": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_mad.png"), -1), (50, 50)
            ),
            "landmarks": [4, 8, 12, 16, 20],
        },
    ],
    "open_hand": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_open_hand.png"), -1),
                (50, 50),
            ),
            "landmarks": [4, 8, 12, 16, 20],
        },
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_sun_with_face.png"), -1),
                (50, 50),
            ),
            "landmarks": [0, 1, 5, 9, 13, 17],
        },
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_dove.png"), -1), (50, 50)
            ),
            "landmarks": [2, 3, 6, 7, 10, 11, 14, 15, 18, 19],
        },
    ],
    "rock_on": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_rock_on.png"), -1),
                (50, 50),
            ),
            "landmarks": [8, 20],
        },
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_fire.png"), -1), (50, 50)
            ),
            "landmarks": [5, 6, 7, 17, 18, 19],
        },
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_crocodile.png"), -1),
                (50, 50),
            ),
            "landmarks": [0],
        },
    ],
    "i_love_you": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_i_love_you.png"), -1),
                (50, 50),
            ),
            "landmarks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20],
        },
    ],
    "peace": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_peace.png"), -1),
                (50, 50),
            ),
            "landmarks": [0, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    ],
    "fist": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_fist.png"), -1), (50, 50)
            ),
            "landmarks": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
        },
    ],
    "thumb_up": [
        {
            "emoji": cv2.resize(
                cv2.imread(os.path.join(emoji_dir_path, "emoji_thumbs_up.png"), -1),
                (50, 50),
            ),
            "landmarks": [0, 1, 2, 3, 4],
        },
    ],
}
