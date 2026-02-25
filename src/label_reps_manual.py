#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import cv2
import pandas as pd


OUTPUT_VIDEOS   = "output_videos"
LABELS_OUT_PATH = os.path.join("rep_level", "rep_labels_matched.csv")


DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 800


os.makedirs(os.path.dirname(LABELS_OUT_PATH), exist_ok=True)


def resize_to_fit(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def parse_rep_video_name(filename):

    if not filename.startswith('pose_'):
        return None, None
    
    name = filename[5:] 
    

    match = re.search(r'_rep(\d+)\.(mp4|avi|mov)$', name, re.IGNORECASE)
    if match:
        rep_id = int(match.group(1))
        source_base = name[:match.start()]
        ext = match.group(2)
        source_name = f"{source_base}.{ext}"
        return source_name, rep_id
    
    return None, None


def scan_output_folder():
    
    reps = []
    
    for filename in sorted(os.listdir(OUTPUT_VIDEOS)):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        
        source, rep_id = parse_rep_video_name(filename)
        if source is not None:
            reps.append({
                'file': source,
                'rep_id': rep_id,
                'rep_video': filename
            })
    
    return pd.DataFrame(reps)


def load_existing_labels():
    if not os.path.exists(LABELS_OUT_PATH):
        return pd.DataFrame(columns=[
            "file", "rep_id", "rep_video",
            "label", "reject_reason",
            "view_score", "flags"
        ])

    df = pd.read_csv(LABELS_OUT_PATH)
    df["rep_id"] = df["rep_id"].astype(int)
    return df


def save_labels(df):
    df = df.drop_duplicates(subset=["file", "rep_id"], keep="last")
    df.to_csv(LABELS_OUT_PATH, index=False)
    print(f"💾 Saved → {LABELS_OUT_PATH}")


def main():
    # Scan actual folder instead of using rep_features.csv
    print("📂 Scanning output_videos folder...")
    df_reps = scan_output_folder()
    df_reps = df_reps.sort_values(["file", "rep_id"]).reset_index(drop=True)
    
    df_labels = load_existing_labels()

    done = set(zip(df_labels["file"], df_labels["rep_id"].astype(int)))

    # Find reps that need labeling
    to_label = []
    for _, row in df_reps.iterrows():
        key = (row["file"], int(row["rep_id"]))
        if key not in done:
            to_label.append(row)
    
    print(f"\n📊 LABELING SUMMARY:")
    print(f"   Videos in folder:       {len(df_reps)}")
    print(f"   Already labelled:       {len(done)}")
    print(f"   ➡️  TO LABEL:            {len(to_label)}")
    print()
    
    if not to_label:
        print("🎉 Nothing to label! All done.")
        return

    cv2.namedWindow("Rep Label Tool", cv2.WINDOW_NORMAL)
    
    labeled_this_session = 0

    for idx, row in enumerate(to_label):
        remaining = len(to_label) - idx
        
        video_path = os.path.join(OUTPUT_VIDEOS, row["rep_video"])

        print("\n" + "=" * 80)
        print(f"📍 REMAINING: {remaining} | Labeled this session: {labeled_this_session}")
        print(f"File   : {row['file']}")
        print(f"Rep ID : {row['rep_id']}")
        print(f"Video  : {video_path}")
        print("=" * 80)

        cap = cv2.VideoCapture(video_path)
        label, reject_reason = None, ""

        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            disp = frame.copy()

            # Show remaining count on video
            cv2.putText(
                disp,
                f"REMAINING: {remaining} | {row['file']} | rep {row['rep_id']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.putText(
                disp,
                "[s]=safe [r]=risky | reject: [1]=not_squat [2]=angle [3]=vis [4]=keep_video | [q]=quit",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

      
            disp = resize_to_fit(disp, DISPLAY_MAX_W, DISPLAY_MAX_H)

            cv2.imshow("Rep Label Tool", disp)
            k = cv2.waitKey(25) & 0xFF

            if k in (ord('s'), ord('S')):
                label = "safe"
                break
            elif k in (ord('r'), ord('R')):
                label = "risky"
                break
            elif k == ord('1'):
                label = "reject"
                reject_reason = "not_squat"
                break
            elif k == ord('2'):
                label = "reject"
                reject_reason = "bad_camera_angle"
                break
            elif k == ord('3'):
                label = "reject"
                reject_reason = "low_visibility"
                break
            elif k == ord('4'):
                label = "reject"
                reject_reason = "keep_video_for_segmentation"
                break
            elif k in (ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                save_labels(df_labels)
                print(f"\n👋 Exiting. Labeled {labeled_this_session} reps this session.")
                return

        cap.release()

        df_labels.loc[len(df_labels)] = {
            "file": row["file"],
            "rep_id": row["rep_id"],
            "rep_video": row["rep_video"],
            "label": label,
            "reject_reason": reject_reason,
            "view_score": None,
            "flags": None,
        }

        done.add((row["file"], int(row["rep_id"])))
        labeled_this_session += 1
        save_labels(df_labels)

    cv2.destroyAllWindows()
    print(f"\n🎉 All done! Labeled {labeled_this_session} reps this session.")
    save_labels(df_labels)


if __name__ == "__main__":
    main()