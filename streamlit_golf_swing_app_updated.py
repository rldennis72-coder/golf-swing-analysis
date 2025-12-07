
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile

st.title("Golf Swing Analysis AI")
st.write("Upload your golf swing video to analyze metrics like club path, face angle, swing plane, and tempo.")

uploaded_file = st.file_uploader("Upload a golf swing video", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(video_path)
    metrics = []

    st.write("Processing video...")

    prev_club_pos = None
    club_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image)
        results_hands = hands.process(image)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Shoulder angle example
            shoulder_angle = np.degrees(np.arctan2(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            ))

            # Estimate club position using wrist midpoint
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            club_pos = ((left_wrist.x + right_wrist.x) / 2, (left_wrist.y + right_wrist.y) / 2)

            # Calculate club path (difference in x positions frame-to-frame)
            club_path_angle = None
            if prev_club_pos:
                dx = club_pos[0] - prev_club_pos[0]
                dy = club_pos[1] - prev_club_pos[1]
                club_path_angle = np.degrees(np.arctan2(dy, dx))
            prev_club_pos = club_pos

            # Face angle approximation using wrist orientation
            face_angle = None
            if results_hands.multi_hand_landmarks:
                # Simplified: difference between wrist and index finger tip
                hand_landmarks = results_hands.multi_hand_landmarks[0].landmark
                wrist = hand_landmarks[0]
                index_tip = hand_landmarks[8]
                face_angle = np.degrees(np.arctan2(index_tip.y - wrist.y, index_tip.x - wrist.x))

            metrics.append({
                "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                "shoulder_angle": shoulder_angle,
                "club_path_angle": club_path_angle,
                "face_angle": face_angle
            })

    cap.release()

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    csv_path = "swing_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    st.write("Analysis complete! Download your results below:")
    st.download_button(label="Download Metrics CSV", data=open(csv_path, "rb"), file_name="swing_metrics.csv")

    # Annotate video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "annotated_swing.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(metrics):
            cv2.putText(frame, f"Shoulder: {metrics[frame_idx]['shoulder_angle']:.1f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if metrics[frame_idx]['club_path_angle']:
                cv2.putText(frame, f"Club Path: {metrics[frame_idx]['club_path_angle']:.1f}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            if metrics[frame_idx]['face_angle']:
                cv2.putText(frame, f"Face Angle: {metrics[frame_idx]['face_angle']:.1f}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    st.download_button(label="Download Annotated Video", data=open(out_path, "rb"), file_name="annotated_swing.mp4")
