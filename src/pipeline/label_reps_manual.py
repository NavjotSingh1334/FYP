#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import re
import cv2
import shutil
import pandas as pd
from datetime import datetime
from collections import defaultdict
from InquirerPy import inquirer


SEGMENTED_FOLDER = "segmented_reps"
SOURCE_VIDEOS_FOLDER = "output_pose_videos" 


LABELS_CSV = os.path.join("rep_level", "rep_labels_merged.csv")
LABELS_OUTPUT_CSV = LABELS_CSV 

MATCH_REPORT_CSV = os.path.join("rep_level", "label_match_report.csv")


BACKUP_DIR = os.path.join("rep_level", "backups")
MAX_BACKUPS = 50  

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 800
FRAME_SKIP = 30



def resize_to_fit(frame, max_w, max_h):
    
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def parse_rep_filename(filename):
    
    match = re.match(r'(.+)_rep(\d+)\.(\w+)$', filename)
    if match:
        base_name = match.group(1)
        rep_num = int(match.group(2))
        ext = match.group(3)
        source_file = f"{base_name}.{ext}"
        return source_file, rep_num
    return None, None


def get_source_video_path(source_file):
   

    pose_name = f"pose_{source_file}"
    pose_path = os.path.join(SOURCE_VIDEOS_FOLDER, pose_name)
    if os.path.exists(pose_path):
        return pose_path

  
    direct_path = os.path.join(SOURCE_VIDEOS_FOLDER, source_file)
    if os.path.exists(direct_path):
        return direct_path

   
    all_videos_path = os.path.join("all_videos", source_file)
    if os.path.exists(all_videos_path):
        return all_videos_path

    return None



def wait_key(paused: bool, play_delay_ms: int) -> int:
    
    if paused:
        return cv2.waitKeyEx(0)
    return cv2.waitKeyEx(play_delay_ms)


def is_left(key: int) -> bool:
   
    return key in (2424832, 81)


def is_right(key: int) -> bool:
    
    return key in (2555904, 83)


def key_to_char(key: int) -> str:
    if 0 <= key <= 255:
        return chr(key).lower()
    return ""



def manual_segment_video(source_file):
    
    video_path = get_source_video_path(source_file)

    if not video_path:
        print(f" Source video not found for: {source_file}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    play_delay = max(1, int(1000.0 / fps))

    current_frame = 0
    paused = True
    pending_start = None
    segments = []

    print(f"\n{'='*60}")
    print(f"  MANUAL SEGMENTATION: {source_file}")
    print(f"  Total frames: {total_frames}")
    print(f"{'='*60}")
    print("  [ = Mark START | ] = Mark END | U = Undo")
    print("  ENTER or P = Done | Q = Cancel")
    print("  SPACE = Pause/Play | LEFT/RIGHT = Skip 30 frames (fallback: A/D or J/L)")
    print(f"{'='*60}\n")

    win_name = "Manual Segmentation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(DISPLAY_MAX_W, 1280), min(DISPLAY_MAX_H, 720))

    while True:
        if total_frames > 0:
            current_frame = max(0, min(total_frames - 1, current_frame))
        else:
            current_frame = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret or frame is None:
            current_frame = 0
            continue

        disp = frame.copy()
        status = "PAUSED" if paused else "PLAYING"

        cv2.putText(disp, f"Frame: {current_frame}/{total_frames} [{status}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, f"Segments: {len(segments)} | Pending START: {pending_start}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(disp, "SPACE Play/Pause | [ START | ] END | U Undo | ENTER/P Done | Q Cancel",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

       
        h, w = disp.shape[:2]
        bar_y = h - 40
        bar_x0, bar_x1 = 50, w - 50
        bar_width = max(1, bar_x1 - bar_x0)

        cv2.rectangle(disp, (bar_x0, bar_y - 10), (bar_x1, bar_y + 10), (50, 50, 50), -1)

        if total_frames > 0:
            for s, e in segments:
                xs = int(bar_x0 + (s / total_frames) * bar_width)
                xe = int(bar_x0 + (e / total_frames) * bar_width)
                cv2.rectangle(disp, (xs, bar_y - 8), (xe, bar_y + 8), (0, 200, 0), -1)

            if pending_start is not None:
                x_start = int(bar_x0 + (pending_start / total_frames) * bar_width)
                cv2.line(disp, (x_start, bar_y - 15), (x_start, bar_y + 15), (0, 255, 255), 3)

            x_cur = int(bar_x0 + (current_frame / total_frames) * bar_width)
            cv2.line(disp, (x_cur, bar_y - 15), (x_cur, bar_y + 15), (0, 0, 255), 2)

        cv2.imshow(win_name, resize_to_fit(disp, DISPLAY_MAX_W, DISPLAY_MAX_H))

        key = wait_key(paused, play_delay)
        ch = key_to_char(key)

        
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            return None

      
        if not paused:
            current_frame += 1
            if total_frames > 0 and current_frame >= total_frames:
                current_frame = 0

        
        if ch == " ":
            paused = not paused

        elif is_left(key) or ch in ("a", "j"):
            current_frame -= FRAME_SKIP
        elif is_right(key) or ch in ("d", "l"):
            current_frame += FRAME_SKIP

        elif ch == "[":
            pending_start = current_frame
            print(f"   ▶ Start marked at frame {current_frame}")

        elif ch == "]":
            if pending_start is None:
                print("   ⚠️ Mark start first with [")
            elif current_frame <= pending_start:
                print("   ⚠️ End must be after start")
            else:
                segments.append((pending_start, current_frame))
                print(f"   ✅ Segment {len(segments)}: frames {pending_start} - {current_frame}")
                pending_start = None

        elif ch == "u":
            if segments:
                removed = segments.pop()
                print(f"   ↩️ Undid segment: {removed}")
            else:
                print("   ⚠️ Nothing to undo")

        
        elif ch in ("\r", "\n", "p"):
            cap.release()
            cv2.destroyWindow(win_name)
            if segments:
                print(f"   ✅ Done! {len(segments)} segments marked")
                return segments
            print("   ⚠️ No segments marked, returning to labeling")
            return None

        elif ch == "q":
            cap.release()
            cv2.destroyWindow(win_name)
            print("   ❌ Cancelled manual segmentation")
            return None


def extract_segment_to_file(source_file, start_frame, end_frame, rep_id):
    
    video_path = get_source_video_path(source_file)
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(source_file)[0]
    output_name = f"{base_name}_rep{rep_id:02d}.mp4"
    output_path = os.path.join(SEGMENTED_FOLDER, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return output_name


def delete_old_reps(source_file):
    
    base_name = os.path.splitext(source_file)[0]
    deleted = []
    for f in os.listdir(SEGMENTED_FOLDER):
        if f.startswith(base_name + "_rep"):
            path = os.path.join(SEGMENTED_FOLDER, f)
            os.remove(path)
            deleted.append(f)
    return deleted


def label_single_rep(rep_video_path, source_file, rep_id, remaining_count):

    cap = cv2.VideoCapture(rep_video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {rep_video_path}")
        return "skip"

    cv2.namedWindow("Rep Label Tool", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        disp = frame.copy()
        cv2.putText(disp, f"REMAINING: {remaining_count} | {source_file} | rep {rep_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, "[S]afe [R]isky | Reject: [1]not_squat [2]angle [3]vis",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, "[M]anual segment | [N]skip | [Q]uit",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Rep Label Tool", resize_to_fit(disp, DISPLAY_MAX_W, DISPLAY_MAX_H))
        key = cv2.waitKey(25) & 0xFF

        if key in (ord("s"), ord("S")):
            cap.release()
            return "safe"
        if key in (ord("r"), ord("R")):
            cap.release()
            return "risky"
        if key == ord("1"):
            cap.release()
            return "reject:not_squat"
        if key == ord("2"):
            cap.release()
            return "reject:bad_camera_angle"
        if key == ord("3"):
            cap.release()
            return "reject:low_visibility"
        if key in (ord("m"), ord("M")):
            cap.release()
            return "manual"
        if key in (ord("n"), ord("N")):
            cap.release()
            return "skip"
        if key in (ord("q"), ord("Q")):
            cap.release()
            return "quit"


def load_labels():
    
    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        return df
    return pd.DataFrame(columns=["file", "rep_id", "rep_video", "label", "reject_reason"])


def _make_backup_if_exists(target_csv: str):
    
    if not os.path.exists(target_csv):
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(target_csv))[0]
    backup_path = os.path.join(BACKUP_DIR, f"{base}_backup_{ts}.csv")

    shutil.copy2(target_csv, backup_path)

   
    backups = sorted(
        [os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) if f.startswith(base + "_backup_") and f.endswith(".csv")]
    )
    if len(backups) > MAX_BACKUPS:
        for old in backups[: len(backups) - MAX_BACKUPS]:
            try:
                os.remove(old)
            except OSError:
                pass


def save_labels(df):
    
    os.makedirs(os.path.dirname(LABELS_OUTPUT_CSV), exist_ok=True)

    keep_cols = ["file", "rep_id", "rep_video", "label", "reject_reason"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""


    df = df[keep_cols].drop_duplicates(subset=["file", "rep_id"], keep="last")


    _make_backup_if_exists(LABELS_OUTPUT_CSV)

   
    tmp_path = LABELS_OUTPUT_CSV + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, LABELS_OUTPUT_CSV)

    print(f"💾 Saved → {LABELS_OUTPUT_CSV} (backup in {BACKUP_DIR}/)")


def main():
    print("\n" + "=" * 60)
    print("  COMPREHENSIVE REP LABELING TOOL (SINGLE LABELS FILE)")
    print("=" * 60)

    if not os.path.isdir(SEGMENTED_FOLDER):
        print(f"❌ Segmented folder not found: {SEGMENTED_FOLDER}")
        return

    df_labels = load_labels()

   
    for col in ["file", "rep_id", "rep_video", "label", "reject_reason"]:
        if col not in df_labels.columns:
            df_labels[col] = ""

   
    labeled_keys = set(zip(df_labels["file"].astype(str), df_labels["rep_id"].astype(int))) if len(df_labels) else set()

    print(f"\n📄 Existing labels (from canonical CSV): {len(df_labels)}")

    rep_files = sorted([f for f in os.listdir(SEGMENTED_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
    print(f"📁 Rep files in folder: {len(rep_files)}")

    reps_by_video = defaultdict(list)
    for rep_file in rep_files:
        source_file, rep_id = parse_rep_filename(rep_file)
        if source_file is not None:
            reps_by_video[source_file].append({"rep_file": rep_file, "rep_id": rep_id, "source_file": source_file})

    for source in reps_by_video:
        reps_by_video[source].sort(key=lambda x: x["rep_id"])

    videos_to_relabel = set()
    if os.path.exists(MATCH_REPORT_CSV):
        df_report = pd.read_csv(MATCH_REPORT_CSV)
        video_col = "video_in_labels" if "video_in_labels" in df_report.columns else "video"
        if video_col in df_report.columns and "status" in df_report.columns:
            mismatched = df_report[df_report["status"] == "mismatch"][video_col].tolist()
            videos_to_relabel = set(mismatched)
            print(f"📊 Videos with count mismatch (will relabel): {len(videos_to_relabel)}")

    mode = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Label unlabeled reps only (continue safely)", "value": "unlabeled"},
            {"name": f"Relabel mismatched videos ({len(videos_to_relabel)} videos)", "value": "mismatch"},
            {"name": "Label ALL reps (overwrite existing labels file)", "value": "all"},
            {"name": "Label specific video", "value": "specific"},
        ],
    ).execute()

    to_label = []

    if mode == "unlabeled":
        for source_file, reps in reps_by_video.items():
            for rep in reps:
                if (source_file, rep["rep_id"]) not in labeled_keys:
                    to_label.append(rep)

    elif mode == "mismatch":
        for source_file in videos_to_relabel:
            if source_file in reps_by_video:
                df_labels = df_labels[df_labels["file"] != source_file]
                labeled_keys = set(zip(df_labels["file"].astype(str), df_labels["rep_id"].astype(int))) if len(df_labels) else set()
                to_label.extend(reps_by_video[source_file])

    elif mode == "all":
        for source_file, reps in reps_by_video.items():
            to_label.extend(reps)
        df_labels = pd.DataFrame(columns=["file", "rep_id", "rep_video", "label", "reject_reason"])
        labeled_keys = set()

    elif mode == "specific":
        video_choices = sorted(reps_by_video.keys())
        selected = inquirer.fuzzy(message="Select video to label:", choices=video_choices).execute()
        if selected in reps_by_video:
            df_labels = df_labels[df_labels["file"] != selected]
            to_label.extend(reps_by_video[selected])

    print(f"\n📊 Reps to label: {len(to_label)}")
    if not to_label:
        print("🎉 Nothing to label!")
        return

    labeled_count = 0
    idx = 0

    while idx < len(to_label):
        rep = to_label[idx]
        remaining = len(to_label) - idx

        rep_path = os.path.join(SEGMENTED_FOLDER, rep["rep_file"])
        if not os.path.exists(rep_path):
            idx += 1
            continue

        print(f"\n📍 {rep['source_file']} | Rep {rep['rep_id']} | Remaining: {remaining}")
        result = label_single_rep(rep_path, rep["source_file"], rep["rep_id"], remaining)

        if result == "quit":
            save_labels(df_labels)
            print(f"\n👋 Exiting. Labeled {labeled_count} reps this session.")
            cv2.destroyAllWindows()
            return

        if result == "skip":
            idx += 1
            continue

        if result == "manual":
            print(f"\n🔧 Starting manual segmentation for: {rep['source_file']}")
            segments = manual_segment_video(rep["source_file"])

            if segments:
                deleted = delete_old_reps(rep["source_file"])
                print(f"   🗑️ Deleted {len(deleted)} old rep files")

                df_labels = df_labels[df_labels["file"] != rep["source_file"]]

                new_reps = []
                for i, (start, end) in enumerate(segments):
                    new_rep_file = extract_segment_to_file(rep["source_file"], start, end, i)
                    if new_rep_file:
                        new_reps.append({"rep_file": new_rep_file, "rep_id": i, "source_file": rep["source_file"]})
                        print(f"   ✅ Created: {new_rep_file}")

                to_label = [r for r in to_label if r["source_file"] != rep["source_file"]]
                for new_rep in reversed(new_reps):
                    to_label.insert(idx, new_rep)
                print(f"   📝 Now label the {len(new_reps)} new reps...")
            else:
                idx += 1
            continue

      
        if result.startswith("reject:"):
            reject_reason = result.split(":", 1)[1]
            new_row = {
                "file": rep["source_file"],
                "rep_id": rep["rep_id"],
                "rep_video": rep["rep_file"],
                "label": "reject",
                "reject_reason": reject_reason,
            }
        else:
            new_row = {
                "file": rep["source_file"],
                "rep_id": rep["rep_id"],
                "rep_video": rep["rep_file"],
                "label": result,
                "reject_reason": "",
            }

        df_labels = pd.concat([df_labels, pd.DataFrame([new_row])], ignore_index=True)
        labeled_count += 1
        idx += 1

   
        if labeled_count % 10 == 0:
            save_labels(df_labels)

    cv2.destroyAllWindows()
    save_labels(df_labels)
    print(f"\n🎉 Done! Labeled {labeled_count} reps this session.")
    print(f"📊 Total labels in canonical CSV: {len(df_labels)}")


if __name__ == "__main__":
    main()
