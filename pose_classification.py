import math
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def classifyPose(landmarks, frame, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        frame: The input frame from the webcam.
        display: A boolean value that if set to true displays the original and classified output images side by side.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    output_image = frame.copy()
    label = 'Unknown Pose'
    color = (0, 0, 255)
    
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    label = 'Warrior II Pose'
                    
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = 'T Pose'

    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            label = 'Tree Pose'
                
    if label != 'Unknown Pose':
        color = (0, 255, 0)  
    
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121); plt.imshow(frame[:,:,::-1]); plt.title("Original Image"); plt.axis('off');
        plt.subplot(122); plt.imshow(output_image[:,:,::-1]); plt.title("Classified Output Image"); plt.axis('off');
        plt.show()
    
    return output_image, label

def detectPoseFromImg(image, pose, display=False):
    '''
    Detects pose landmarks from an image using MediaPipe Pose.
    
    Args:
        image: Input image in BGR format.
        pose: MediaPipe Pose object.
        display: Boolean indicating whether to display the images.
        
    Returns:
        output_image: The image with pose landmarks drawn.
        landmarks: List of pose landmarks.
    '''
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5)
        connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              int(landmark.z * width)))
    
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121); plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off');
        plt.subplot(122); plt.imshow(output_image[:,:,::-1]); plt.title("Output Image"); plt.axis('off');
        plt.show()
    
    return output_image, landmarks

def process_vid_classification(pose_video):
    '''
    Processes video input from the webcam and performs pose detection and classification.
    
    Args:
        pose_video: MediaPipe Pose object configured for video processing.
        
    Returns:
        None
    '''
    camera_video = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not camera_video.isOpened():
        print("Error: Unable to access the webcam!")
        return
    
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
    
    while camera_video.isOpened():
        ok, frame = camera_video.read()
        
        if not ok:
            print("Error: Unable to read frame from webcam!")
            break
        
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        frame, landmarks = detectPoseFromImg(frame, pose_video, display=False)
        
        if landmarks:
            frame, _ = classifyPose(landmarks, frame, display=False)
        
        cv2.imshow('Pose Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    
    camera_video.release()
    cv2.destroyAllWindows()

# # Main execution
# if __name__ == "__main__":
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_video:
#         process_vid_classification(pose_video)