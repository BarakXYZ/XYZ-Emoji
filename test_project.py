# ~ BarakXYZ - XYZ Emoji - 2024 - CS50P Final Project ~

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
from project import play_sound, release_exit, calculate_distance, overlay_emoji


def test_play_sound(mocker, capsys):
    mocker.patch("os.path.exists", return_value=False)
    play_sound("nonexistent_file.wav")
    captured = capsys.readouterr()
    assert "The file 'nonexistent_file.wav' does not exist." in captured.out


def test_play_sound_invalid_volume(mocker, capsys):
    mocker.patch("os.path.exists", return_value=True)
    play_sound("existent_file.wav", volume=1.5)
    captured = capsys.readouterr()
    assert "Volume must be between 0.0 and 1.0." in captured.out


def test_release_exit(mocker, capsys):
    # Mock external dependencies
    cap_mock = Mock()
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("sys.exit")

    # Call the function
    release_exit(cap_mock, "Goodbye")

    # Capture print statements
    captured = capsys.readouterr()

    # Assertions
    cap_mock.release.assert_called_once()
    assert "Goodbye" in captured.out
    sys.exit.assert_called_once_with("0")


def test_release_exit_failure(mocker, capsys):
    # Mock external dependencies
    cap_mock = Mock()
    cap_mock.release.side_effect = Exception("Camera Error")
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("sys.exit")

    # Call the function
    release_exit(cap_mock, "Goodbye")

    # Capture print statements
    captured = capsys.readouterr()

    # Assertions
    cap_mock.release.assert_called_once()
    assert "Couldn't close and release the camera capture: Camera Error" in captured.out
    sys.exit.assert_called_once_with(1)


def test_calculate_distance():
    # Create mock objects for hand landmarks and mediapipe hands
    hand_lmks = MagicMock()
    mp_hands = MagicMock()

    # Simulate landmark data for thumb tip and index finger tip
    thumb_tip_mock = MagicMock(x=1, y=1)
    index_tip_mock = MagicMock(x=4, y=5)

    hand_lmks.landmark = MagicMock()
    hand_lmks.landmark.__getitem__.side_effect = [
        thumb_tip_mock,
        index_tip_mock,
    ]  # Simulate accessing landmarks by index

    mp_hands.HandLandmark.THUMB_TIP = 0  # Mock index for thumb tip
    mp_hands.HandLandmark.INDEX_FINGER_TIP = 1  # Mock index for index finger tip

    # Call the function under test
    distance = calculate_distance(hand_lmks, mp_hands)

    # Expected distance calculation
    expected_distance = np.sqrt((4 - 1) ** 2 + (5 - 1) ** 2)

    # Assert the calculated distance is as expected
    assert distance == expected_distance


@pytest.fixture
def sample_image():
    # Create a sample image of dimensions 100x100 with 3 RGB channels, filled with zeros
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_emoji():
    # Create a sample emoji of dimensions 10x10 with 4 channels (RGBA), filled with ones
    emoji = np.ones((10, 10, 4), dtype=np.uint8)
    emoji[:, :, 3] = 255  # Set the alpha channel to fully opaque
    return emoji


def test_overlay_emoji(sample_image, sample_emoji):
    img_dim = sample_image.shape
    position = (50, 50)  # Center of the image
    overlay_emoji(sample_image, img_dim, position, sample_emoji)

    # Assert that the center of the image now contains the emoji
    # Check a pixel in the center of where the emoji should be
    center_of_emoji = (50, 50)
    expected_color = sample_emoji[5, 5, :3]  # Ignore the alpha channel
    assert np.array_equal(sample_image[center_of_emoji], expected_color)
