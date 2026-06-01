# ==============================================================================
# face_identity.py — FaceNet Embedding Extraction for Identity Matching
# ==============================================================================
# Provides:
#   get_facenet()          - Singleton FaceNet (InceptionResnetV1) loader
#   get_identity_mtcnn()   - Singleton MTCNN (keep_all=True) loader
#   detect_and_embed()     - Detect all faces + extract 512-d embeddings
#   cosine_similarity()    - Standard cosine similarity between two embeddings
#
# All operations are in-memory. No disk I/O.
# Thread-safe: singletons are loaded once.
# ==============================================================================

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# SINGLETONS (lazy-loaded)
# ==============================================================================

_facenet = None
_id_mtcnn = None


def get_facenet():
    """
    Lazy-load InceptionResnetV1 with VGGFace2 pretrained weights.
    Returns a frozen model in eval mode on DEVICE.
    """
    global _facenet
    if _facenet is None:
        from facenet_pytorch import InceptionResnetV1
        _facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        print(f"[INFO] FaceNet loaded on {DEVICE}")
    return _facenet


def get_identity_mtcnn():
    """
    Lazy-load MTCNN configured for multi-face detection.
    keep_all=True to return ALL detected faces, not just the primary.
    """
    global _id_mtcnn
    if _id_mtcnn is None:
        from facenet_pytorch import MTCNN
        _id_mtcnn = MTCNN(
            image_size=160,         # FaceNet native input size
            margin=20,
            min_face_size=40,
            keep_all=True,          # Detect ALL faces
            thresholds=[0.6, 0.7, 0.7],
            device=DEVICE,
            post_process=False      # Return raw pixel tensors
        )
        print("[INFO] Identity MTCNN loaded (keep_all=True)")
    return _id_mtcnn


# ==============================================================================
# DETECTION + EMBEDDING
# ==============================================================================

def detect_and_embed(pil_image: Image.Image) -> List[Dict]:
    """
    Detect all faces in a PIL image and extract FaceNet embeddings.

    Returns a list of dicts, each containing:
        - "box": [x1, y1, x2, y2] (int coordinates)
        - "embedding": torch.Tensor of shape (512,)
        - "prob": detection probability

    All operations in-memory. No disk I/O.

    Args:
        pil_image: RGB PIL Image

    Returns:
        List of face dicts, or empty list if no faces found.
    """
    mtcnn = get_identity_mtcnn()
    facenet = get_facenet()

    # Detect all faces — returns tensors and boxes
    boxes, probs = mtcnn.detect(pil_image)

    if boxes is None or len(boxes) == 0:
        return []

    faces = []
    img_w, img_h = pil_image.size

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None or prob < 0.5:
            continue

        # Clamp box to image bounds
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop face from PIL image
        face_crop = pil_image.crop((x1, y1, x2, y2))
        # Resize to FaceNet input size (160x160)
        face_crop = face_crop.resize((160, 160), Image.LANCZOS)

        # Convert to tensor: (3, 160, 160), normalized to [-1, 1]
        face_tensor = torch.from_numpy(
            np.array(face_crop, dtype=np.float32)
        ).permute(2, 0, 1)  # HWC → CHW
        face_tensor = (face_tensor - 127.5) / 128.0  # FaceNet normalization
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 160, 160)

        # Extract embedding
        with torch.no_grad():
            embedding = facenet(face_tensor).squeeze(0)  # (512,)

        faces.append({
            "box": [x1, y1, x2, y2],
            "embedding": embedding.cpu(),
            "prob": float(prob)
        })

    return faces


# ==============================================================================
# SIMILARITY
# ==============================================================================

def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        emb_a: Tensor of shape (512,)
        emb_b: Tensor of shape (512,)

    Returns:
        float in range [-1, 1]. Higher = more similar.
    """
    emb_a = emb_a.float().flatten()
    emb_b = emb_b.float().flatten()

    dot = torch.dot(emb_a, emb_b)
    norm_a = torch.norm(emb_a)
    norm_b = torch.norm(emb_b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(dot / (norm_a * norm_b))
