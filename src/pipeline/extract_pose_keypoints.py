#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from InquirerPy import inquirer

OUTPUT_DIR = os.path.join("frame_level")
KEYPOINTS_CSV = os.path.join(OUTPUT_DIR, "pose_keypoints.csv")
KEYPOINTS_RAW_CSV = os.path.join(OUTPUT_DIR, "pose_keypoints_raw.csv")
POSE_VIDEOS_DIR = "output_pose_videos"


RTMPOSE_SIZE = 'm'


USE_WHOLEBODY = False

CONFIDENCE_THRESHOLD = 0.3

OUTPUT_POSE_VIDEOS = True

BACKEND = 'rtmlib'


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(POSE_VIDEOS_DIR, exist_ok=True)


# COCO 17 keypoints (body only)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Joint indices for squat analysis
JOINT_IDX = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

# ==================================================


def choose_input_folder():
    """Let user select input folder."""
    cwd = os.getcwd()
    choices = []
    
    for name in ['videos', 'all_videos', 'raw_videos', 'input_videos', 'rep_seg_vids']:
        path = os.path.join(cwd, name)
        if os.path.isdir(path):
            choices.append((name, path))
    
    for name in sorted(os.listdir(cwd)):
        full = os.path.join(cwd, name)
        if os.path.isdir(full) and full not in [c[1] for c in choices]:
            if not name.startswith('.') and name not in ['__pycache__', 'models', 'frame_level', 'rep_level', 'output_videos', 'output_pose_videos']:
                choices.append((name, full))
    
    if not choices:
        print("⚠️ No subfolders found. Using current directory.")
        return cwd
    
    return inquirer.select(
        message="📁 Select folder containing squat videos:",
        choices=[{"name": n, "value": p} for n, p in choices]
    ).execute()


def angle_between_points(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3. Returns degrees or NaN."""
    if any(p is None for p in [p1, p2, p3]):
        return np.nan
    
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return np.nan
    
    cos_angle = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def trunk_lean_angle(shoulder, hip):
    """Calculate trunk lean from vertical. 0 = upright, positive = forward lean."""
    if shoulder is None or hip is None:
        return np.nan
    
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return np.nan
    
    angle = np.degrees(np.arctan2(dx, -dy))
    return abs(angle)


def get_point_if_valid(keypoints, scores, idx, threshold=0.3):
    """Get (x, y) if confidence above threshold, else None."""
    if idx >= len(keypoints) or idx >= len(scores):
        return None
    if scores[idx] < threshold:
        return None
    return (float(keypoints[idx][0]), float(keypoints[idx][1]))


def draw_skeleton(frame, keypoints, scores, threshold=0.3):
    """Draw skeleton overlay on frame."""
    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            if scores[i] >= threshold and scores[j] >= threshold:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i in range(min(len(keypoints), 17)):
        if scores[i] >= threshold:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            # Red for lower body (important for squats)
            color = (0, 0, 255) if i in [11, 12, 13, 14, 15, 16] else (255, 0, 0)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
    
    return frame


def compute_biomechanical_features(keypoints, scores, img_height):
    
    "Compute biomechanical features from keypoints."
    features = {}
    
    # Get joints
    l_shoulder = get_point_if_valid(keypoints, scores, 5, CONFIDENCE_THRESHOLD)
    r_shoulder = get_point_if_valid(keypoints, scores, 6, CONFIDENCE_THRESHOLD)
    l_hip = get_point_if_valid(keypoints, scores, 11, CONFIDENCE_THRESHOLD)
    r_hip = get_point_if_valid(keypoints, scores, 12, CONFIDENCE_THRESHOLD)
    l_knee = get_point_if_valid(keypoints, scores, 13, CONFIDENCE_THRESHOLD)
    r_knee = get_point_if_valid(keypoints, scores, 14, CONFIDENCE_THRESHOLD)
    l_ankle = get_point_if_valid(keypoints, scores, 15, CONFIDENCE_THRESHOLD)
    r_ankle = get_point_if_valid(keypoints, scores, 16, CONFIDENCE_THRESHOLD)
    
    # Determine better side
    left_conf = sum([scores[i] for i in [5, 11, 13, 15] if i < len(scores)])
    right_conf = sum([scores[i] for i in [6, 12, 14, 16] if i < len(scores)])
    use_left = left_conf >= right_conf
    features['use_left'] = int(use_left)
    
    # Select primary side
    if use_left:
        shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle
    else:
        shoulder, hip, knee, ankle = r_shoulder, r_hip, r_knee, r_ankle
    
    # Hip Y (normalized) - REQUIRED for LSTM segmenter
    if hip is not None:
        features['hip_y'] = hip[1] / img_height
    else:
        features['hip_y'] = np.nan
    
    # Knee angle (hip-knee-ankle) - REQUIRED for LSTM segmenter
    features['knee_deg'] = angle_between_points(hip, knee, ankle)
    
    # Hip angle (shoulder-hip-knee)
    features['hip_deg'] = angle_between_points(shoulder, hip, knee)
    
    # Trunk lean (forward tilt from vertical)
    features['trunk_deg'] = trunk_lean_angle(shoulder, hip)
    
    # Ankle angle estimate
    if knee is not None and ankle is not None:
        foot_proxy = (ankle[0], ankle[1] + 0.05 * img_height)
        features['ankle_deg'] = angle_between_points(knee, ankle, foot_proxy)
    else:
        features['ankle_deg'] = np.nan
    
    # Validity
    valid_joints = sum([1 for v in [shoulder, hip, knee, ankle] if v is not None])
    features['valid'] = int(valid_joints >= 3)
    
    return features


class RTMLibExtractor:
    
    def __init__(self, model_size='m', wholebody=False):
        try:
            from rtmlib import Wholebody, Body
        except ImportError:
            raise ImportError(
                "rtmlib not installed. Install with:\n"
                "pip install rtmlib"
            )
        
        print(f"🧠 Loading RTMPose-{model_size} ({'wholebody' if wholebody else 'body'})...")
        
        # Map size to model
        mode = 'balanced'  # 'performance', 'balanced', 'lightweight'
        if model_size == 'l':
            mode = 'performance'
        elif model_size == 's' or model_size == 't':
            mode = 'lightweight'
        
        if wholebody:
            self.model = Wholebody(mode=mode, to_openpose=False)
        else:
            self.model = Body(mode=mode, to_openpose=False)
        
        self.wholebody = wholebody
        print("✅ RTMPose loaded")
    
    def extract(self, frame):
        """Extract pose. Returns keypoints [N, 2] and scores [N]."""
        try:
            keypoints, scores = self.model(frame)
            
            if keypoints is None or len(keypoints) == 0:
                return None, None
            
            # Take first person
            kpts = keypoints[0] if len(keypoints.shape) == 3 else keypoints
            scrs = scores[0] if len(scores.shape) == 2 else scores
            
            # Ensure we have at least 17 keypoints
            if len(kpts) < 17:
                return None, None
            
            return kpts[:17], scrs[:17]  # Return only body keypoints
            
        except Exception as e:
            return None, None


class MMPoseExtractor:
    
    "Full MMPose extractor (more options but harder setup)."
    
    
    def __init__(self, model_size='m'):
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError:
            raise ImportError(
                "MMPose not installed. Install with:\n"
                "pip install openmim\n"
                "mim install mmengine mmcv mmdet mmpose"
            )
        
        print(f"🧠 Loading MMPose RTMPose-{model_size}...")
        
        # RTMPose model
        self.inferencer = MMPoseInferencer(
            pose2d=f'rtmpose-{model_size}',
            device='cuda:0' if self._has_cuda() else 'cpu'
        )
        print("✅ MMPose loaded")
    
    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def extract(self, frame):
        "Extract pose from frame."
        try:
            result = next(self.inferencer(frame, return_vis=False))
            
            if not result['predictions'] or len(result['predictions'][0]) == 0:
                return None, None
            
            pred = result['predictions'][0][0]  # First person
            keypoints = pred['keypoints']
            scores = pred['keypoint_scores']
            
            return np.array(keypoints), np.array(scores)
            
        except Exception as e:
            return None, None


def process_video(video_path, extractor, output_video=True):
    "Process video and extract pose keypoints."
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return [], []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fname = os.path.basename(video_path)
    frame_data = []
    raw_data = []
    
    # Video writer
    video_writer = None
    if output_video:
        out_path = os.path.join(POSE_VIDEOS_DIR, f"pose_{fname}")
        video_writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"  {fname[:30]}", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract pose
        keypoints, scores = extractor.extract(frame)
        
        if keypoints is not None and scores is not None:
            # Compute features
            features = compute_biomechanical_features(keypoints, scores, height)
            features['file'] = fname
            features['frame'] = frame_idx
            frame_data.append(features)
            
            # Raw keypoints for GCN
            raw_entry = {'file': fname, 'frame': frame_idx}
            for i, name in enumerate(COCO_KEYPOINTS):
                if i < len(keypoints):
                    raw_entry[f'{name}_x'] = keypoints[i][0]
                    raw_entry[f'{name}_y'] = keypoints[i][1]
                    raw_entry[f'{name}_conf'] = scores[i] if i < len(scores) else 0
            raw_data.append(raw_entry)
            
            # Draw skeleton
            if video_writer:
                frame_vis = draw_skeleton(frame.copy(), keypoints, scores)
                cv2.putText(frame_vis, f"Frame: {frame_idx}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if not np.isnan(features.get('knee_deg', np.nan)):
                    cv2.putText(frame_vis, f"Knee: {features['knee_deg']:.1f} deg", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if not np.isnan(features.get('hip_y', np.nan)):
                    cv2.putText(frame_vis, f"Hip Y: {features['hip_y']:.3f}", (10, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                video_writer.write(frame_vis)
        else:
            # No detection
            frame_data.append({
                'file': fname, 'frame': frame_idx,
                'hip_y': np.nan, 'knee_deg': np.nan,
                'hip_deg': np.nan, 'trunk_deg': np.nan,
                'ankle_deg': np.nan, 'valid': 0
            })
            
            if video_writer:
                cv2.putText(frame, f"Frame: {frame_idx} - No detection", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                video_writer.write(frame)
        
        pbar.update(1)
        frame_idx += 1
    
    pbar.close()
    cap.release()
    
    if video_writer:
        video_writer.release()
        print(f"   🎬 Saved: {POSE_VIDEOS_DIR}/pose_{fname}")
    
    return frame_data, raw_data


def main():
    print("="*70)
    print("🏋️ POSE KEYPOINT EXTRACTION (RTMPose)")
    print("="*70)
    
    # Select folder
    input_folder = choose_input_folder()
    if not input_folder or not os.path.isdir(input_folder):
        print("❌ Invalid folder")
        return
    
    # Find videos
    videos = [f for f in os.listdir(input_folder)
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"❌ No videos found in {input_folder}")
        return
    
    print(f"📁 Folder: {input_folder}")
    print(f"🎞 Found {len(videos)} videos")
    print(f"⚙️ Model: RTMPose-{RTMPOSE_SIZE}")
    print(f"⚙️ Backend: {BACKEND}")
    
    # Initialize extractor
    try:
        if BACKEND == 'rtmlib':
            extractor = RTMLibExtractor(RTMPOSE_SIZE, USE_WHOLEBODY)
        else:
            extractor = MMPoseExtractor(RTMPOSE_SIZE)
    except ImportError as e:
        print(f"\n❌ {e}")
        print("\nFalling back to YOLOv8-Pose...")
        try:
            from ultralytics import YOLO
            
            class YOLOExtractor:
                def __init__(self):
                    self.model = YOLO('yolov8m-pose.pt')
                    print("✅ YOLOv8-Pose loaded (fallback)")
                
                def extract(self, frame):
                    results = self.model(frame, verbose=False)
                    if len(results) == 0 or results[0].keypoints is None:
                        return None, None
                    kpts = results[0].keypoints
                    if kpts.xy is None or len(kpts.xy) == 0:
                        return None, None
                    keypoints = kpts.xy[0].cpu().numpy()
                    scores = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(17)
                    return keypoints, scores
            
            extractor = YOLOExtractor()
        except ImportError:
            print("❌ No pose estimation backend available!")
            print("   Install one of: pip install rtmlib / pip install ultralytics")
            return
    
    # Process videos
    all_frame_data = []
    all_raw_data = []
    
    # Check which videos already have pose videos (skip if already processed)
    skipped_videos = []
    videos_to_process = []
    
    for video_name in videos:
        pose_video_path = os.path.join(POSE_VIDEOS_DIR, f"pose_{video_name}")
        base_name = os.path.splitext(video_name)[0]
        pose_video_mp4 = os.path.join(POSE_VIDEOS_DIR, f"pose_{base_name}.mp4")
        pose_video_avi = os.path.join(POSE_VIDEOS_DIR, f"pose_{base_name}.avi")
        
        if os.path.exists(pose_video_path) or os.path.exists(pose_video_mp4) or os.path.exists(pose_video_avi):
            skipped_videos.append(video_name)
        else:
            videos_to_process.append(video_name)
    
    if skipped_videos:
        print(f"⏭️  Skipping {len(skipped_videos)} videos (pose videos already exist)")
        print(f"   To re-process, delete files in {POSE_VIDEOS_DIR}/")
    
    if not videos_to_process:
        print(f"\n⚠️  All videos already processed. Nothing to do.")
        print(f"   Delete pose videos in {POSE_VIDEOS_DIR}/ to re-process.")
        return
    
    print(f"\n🔄 Processing {len(videos_to_process)} videos...\n")
    
    for video_name in videos_to_process:
        video_path = os.path.join(input_folder, video_name)
        print(f"🎥 {video_name}")
        
        frame_data, raw_data = process_video(video_path, extractor, OUTPUT_POSE_VIDEOS)
        
        all_frame_data.extend(frame_data)
        all_raw_data.extend(raw_data)
        
        print(f"   ✅ {len(frame_data)} frames")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    df_features_new = pd.DataFrame(all_frame_data)
    
    # Get list of new files we just processed
    new_files = set(df_features_new['file'].unique())
    
    # Append to existing CSV if it exists (remove duplicates by file)
    if os.path.exists(KEYPOINTS_CSV):
        df_existing = pd.read_csv(KEYPOINTS_CSV)
        # Remove any existing entries for files we just processed (to avoid duplicates)
        df_existing = df_existing[~df_existing['file'].isin(new_files)]
        df_features = pd.concat([df_existing, df_features_new], ignore_index=True)
        print(f"   • Appended to existing {KEYPOINTS_CSV}")
        print(f"     - Existing (other files): {len(df_existing)} rows")
        print(f"     - New: {len(df_features_new)} rows")
        print(f"     - Total: {len(df_features)} rows")
    else:
        df_features = df_features_new
        print(f"   • Created {KEYPOINTS_CSV} ({len(df_features)} rows)")
    
    df_features.to_csv(KEYPOINTS_CSV, index=False)
    
    if all_raw_data:
        df_raw_new = pd.DataFrame(all_raw_data)
        
        if os.path.exists(KEYPOINTS_RAW_CSV):
            df_raw_existing = pd.read_csv(KEYPOINTS_RAW_CSV)
            df_raw_existing = df_raw_existing[~df_raw_existing['file'].isin(new_files)]
            df_raw = pd.concat([df_raw_existing, df_raw_new], ignore_index=True)
            print(f"   • Appended to existing {KEYPOINTS_RAW_CSV}")
            print(f"     - Total: {len(df_raw)} rows")
        else:
            df_raw = df_raw_new
            print(f"   • Created {KEYPOINTS_RAW_CSV} ({len(df_raw)} rows)")
        
        df_raw.to_csv(KEYPOINTS_RAW_CSV, index=False)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Videos in folder: {len(videos)}")
    print(f"   Skipped (already processed): {len(skipped_videos)}")
    print(f"   Processed this run: {len(videos_to_process)}")
    print(f"   Frames extracted: {len(all_frame_data)}")
    
    if all_frame_data:
        valid = sum(1 for f in all_frame_data if f.get('valid', 0))
        hip_valid = sum(1 for f in all_frame_data if not np.isnan(f.get('hip_y', np.nan)))
        knee_valid = sum(1 for f in all_frame_data if not np.isnan(f.get('knee_deg', np.nan)))
        
        print(f"   Valid frames: {valid/len(all_frame_data)*100:.1f}%")
        print(f"   hip_y coverage: {hip_valid/len(all_frame_data)*100:.1f}%")
        print(f"   knee_deg coverage: {knee_valid/len(all_frame_data)*100:.1f}%")
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()