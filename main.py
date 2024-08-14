import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from img_pose_detection_utils import detectPoseFromImg
from vid_pose_detection import process_video
from pose_classification import calculateAngle, classifyPose, process_vid_classification

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Main execution
if __name__ == "__main__":
    # Uncomment the following lines if you want to perform pose detection on an image
    '''
    # Load the image
    sample_img = cv2.imread('media/Sample2.jpg')

    if sample_img is None:
        print("Error: Image not found or unable to read! Check your folder")
    else:
        # Call the pose detection function
        detectPoseFromImg(sample_img, pose, display=True)
    '''

    # Uncomment the following lines if you want to perform pose detection on a video
    '''
    video_path = 0  # Use 0 for webcam or provide the path to a video file

    # Call the function to process the video
    process_video(video_path, pose)
    '''

    # Call the function to perform pose classification using the webcam
    process_vid_classification(pose_video)

    # Clean up
    cv2.destroyAllWindows()