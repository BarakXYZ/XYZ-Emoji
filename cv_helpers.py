# ~ BarakXYZ - XYZ Emoji - 2024 - CS50P Final Project ~

import cv2
import time


# Function to calculate the frames per second (FPS)
def calculate_fps(previous_time, img):
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(
        img, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    return current_time


# Function to find the index of the camera
def find_device_index(
    start_index=0, end_index=10, find_all_devices=False, return_index=False
):
    for index in range(start_index, end_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Found video capture device at index: {index}")
            if return_index:
                return index
            cap.release()
            if find_all_devices == False:
                break
        else:
            print(f"No device found at index: {index}")


def main():
    # Use the DirectShow backend
    cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
    else:
        # Continuously capture frames from the camera
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break  # Exit the loop if there's a problem reading a frame

            # Display the captured frame
            cv2.imshow("Frame", frame)
            # If 'q' is pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    find_device_index(find_all_devices=False)
