import cv2
import time
from img_pose_detection_utils import detectPoseFromImg  # Import the function from the file where you defined it

def process_video(video_path, pose, display_window_name='Pose Detection'):
    '''
    Processes a video file or webcam feed and performs pose detection on each frame.
    
    Args:
        video_path: Path to the video file or use 0 for webcam.
        pose: The pose setup function required for pose detection.
        display_window_name: Name of the window to display the video feed.
        
    Returns:
        None
    '''
    # Initialize VideoCapture object
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Video file not found or unable to open!")
        return
    
    # Initialize a variable to store the time of the previous frame
    time1 = 0
    
    while video.isOpened():
        # Read a frame
        ok, frame = video.read()
        
        if not ok:
            break
        
        # Flip the frame horizontally for natural (selfie-view) visualization
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape
        
        # Resize the frame while keeping the aspect ratio
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform pose landmark detection
        frame, _ = detectPoseFromImg(frame, pose, display=False)
        
        # Set the time for this frame to the current time
        time2 = time.time()
        
        # Calculate frames per second
        if (time2 - time1) > 0:
            frames_per_second = 1.0 / (time2 - time1)
            cv2.putText(frame, f'FPS: {int(frames_per_second)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        time1 = time2
        
        # Display the frame
        cv2.imshow(display_window_name, frame)
        
        # Break the loop if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release the VideoCapture object and close windows
    video.release()
    cv2.destroyAllWindows()
