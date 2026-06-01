import cv2
import os
import numpy as np


def extract_and_save_frames(
    video_path,
    output_dir,
    num_frames=32,
    img_size=224
):
    """
    Extracts exactly `num_frames` frames uniformly from a video
    and saves them as JPG images.
    """

    # --------------------------------------------------
    # Resolve and create output directory (CRITICAL)
    # --------------------------------------------------
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(output_dir):
        raise RuntimeError(f"Invalid output directory: {output_dir}")

    # --------------------------------------------------
    # Open video
    # --------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read frames from video: {video_path}")

    # --------------------------------------------------
    # Select frame indices uniformly
    # --------------------------------------------------
    frame_indices = np.linspace(
        0, total_frames - 1, num_frames, dtype=int
    )
    frame_indices = set(frame_indices)

    saved_count = 0
    current_frame = 0

    # --------------------------------------------------
    # Read & save frames
    # --------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            frame = cv2.resize(frame, (img_size, img_size))
            frame_path = os.path.join(
                output_dir,
                f"frame_{saved_count:04d}.jpg"
            )

            success = cv2.imwrite(frame_path, frame)
            if not success:
                cap.release()
                raise RuntimeError(f"Failed to write frame: {frame_path}")

            saved_count += 1

            if saved_count >= num_frames:
                break

        current_frame += 1

    cap.release()

    if saved_count == 0:
        raise RuntimeError("No frames were saved from the video")

    print(f"[INFO] Saved {saved_count} frames to {output_dir}")
