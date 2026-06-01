import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN


def crop_faces_from_frames(
    frames_root,
    faces_root,
    image_size=224,
    device=None
):
    """
    Detects and crops faces from saved frames using MTCNN.
    Saves proper RGB face images (NOT black).
    
    NOTE: This is for BATCH processing with nested directory structure:
    frames_root/label/video_name/frames
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        image_size=image_size,
        margin=20,
        keep_all=False,
        device=device
    )

    for label in os.listdir(frames_root):
        label_path = os.path.join(frames_root, label)
        if not os.path.isdir(label_path):
            continue

        for video_name in os.listdir(label_path):
            video_frame_dir = os.path.join(label_path, video_name)
            output_video_dir = os.path.join(faces_root, label, video_name)

            os.makedirs(output_video_dir, exist_ok=True)

            for frame_file in os.listdir(video_frame_dir):
                if not frame_file.lower().endswith(".jpg"):
                    continue

                frame_path = os.path.join(video_frame_dir, frame_file)
                save_path = os.path.join(output_video_dir, frame_file)

                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception:
                    continue

                # ✅ THIS LINE FIXES EVERYTHING
                mtcnn(img, save_path=save_path)

            print(f"[INFO] Faces saved for {label}/{video_name}")


def crop_single_video_faces(
    frames_dir: str,
    faces_dir: str,
    image_size: int = 224,
    device: str = None
) -> int:
    """
    Crop faces from a single video's frames directory.
    
    This is used by the inference pipeline for single video processing.
    
    Args:
        frames_dir: Directory containing frame JPGs (flat structure)
        faces_dir: Output directory for cropped faces
        image_size: Output face image size (default 224)
        device: 'cuda' or 'cpu'
        
    Returns:
        count: Number of faces successfully cropped
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    mtcnn = MTCNN(
        image_size=image_size,
        margin=20,
        keep_all=False,
        device=device
    )
    
    os.makedirs(faces_dir, exist_ok=True)
    
    count = 0
    
    for frame_file in sorted(os.listdir(frames_dir)):
        if not frame_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        frame_path = os.path.join(frames_dir, frame_file)
        save_path = os.path.join(faces_dir, frame_file)
        
        try:
            img = Image.open(frame_path).convert("RGB")
            face = mtcnn(img, save_path=save_path)
            
            if face is not None:
                count += 1
        except Exception as e:
            print(f"[WARN] Face detection failed for {frame_file}: {e}")
            continue
    
    print(f"[INFO] Cropped {count} faces from {frames_dir}")
    return count

