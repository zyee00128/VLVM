import numpy as np
import torch
from vlfm.vlm.detections import ObjectDetections

def _build_vqa_question(phrase: str, _vqa_prompt:str) -> str:
    """Construct the BLIP2 VQA question.

    Question: Is this a {obj}? 
    Answer: (multi-class takes the primary class).
    """
    primary = phrase.split("|")[0].strip()
    question = f"Question: {_vqa_prompt}"
    if not primary.endswith("ing"):
        question += "a "
    question += primary + "? Answer:"
    return question

def vqa_confirm_detections(
    detections: ObjectDetections,
    use_vqa,
    vqa_client,
    vqa_prompt: str,
) -> None:
    """BLIP2 VQA false-positive confirmation.

    Projects each TSP3D 3D box onto the RGB image, crops the box pixels,
    asks BLIP2 'Is this a {obj}?', and drops (in-place) detections whose
    answer does not start with 'yes' so they never reach target memory.
    """
    if not use_vqa or vqa_client is None or detections.num_detections == 0:
        return
    # Server-unavailable guard: skip VQA entirely (no 10x/20-30s retry + exit()
    # in send_request); detections pass through unconfirmed.
    if not getattr(vqa_client, "is_available", lambda: True)():
        print(
            "[VQA] BLIP2 VQA server unavailable — skipping confirmation "
            "(detections pass through unconfirmed).",
            flush=True,
        )
        return
    rgb = detections.image_source
    tf = detections.tf_camera_to_episodic
    fx, fy = detections.fx, detections.fy
    if rgb is None or tf is None or fx is None or fy is None:
        return
        
    H, W = rgb.shape[:2]
    tf_epi_to_cam = np.linalg.inv(np.asarray(tf, dtype=np.float64))

    keep = []
    for idx, phrase in enumerate(detections.phrases):
        box_np = detections.boxes[idx].cpu().numpy()  # (8, 3) world frame
        pts_world_homo = np.hstack([box_np, np.ones((8, 1))])
        # Inverse camera->episodic gives the camera frame in the SAME convention
        # used by _project_rgbd_to_3d_point_cloud: [forward, right, down].
        # Depth is along forward (x), right is horizontal (y), down is vertical (z).
        pts_cam = (tf_epi_to_cam @ pts_world_homo.T).T[:, :3]
        depth = pts_cam[:, 0]
        # Box crossing/behind the camera plane cannot be visually confirmed -> drop.
        if np.any(depth <= 0.1):
            keep.append(False)
            continue

        # Perspective projection: u uses right (horizontal), v uses down (image v grows downward).
        u = pts_cam[:, 1] * fx / depth + W / 2.0
        v = pts_cam[:, 2] * fy / depth + H / 2.0
        pad = 8
        u0 = int(max(0, np.floor(u.min()) - pad))
        u1 = int(min(W, np.ceil(u.max()) + pad))
        v0 = int(max(0, np.floor(v.min()) - pad))
        v1 = int(min(H, np.ceil(v.max()) + pad))
        if u1 - u0 < 8 or v1 - v0 < 8:
            keep.append(False)
            continue

        crop = rgb[v0:v1, u0:u1]
        question = _build_vqa_question(phrase, _vqa_prompt=vqa_prompt)
        try:
            answer = vqa_client.ask(crop, question)
        except Exception as e:
            print(
                f"[VQA] VQA request failed ({e}) — keeping current and remaining "
                "detections unconfirmed.",
                flush=True,
            )
            keep.append(True)
            keep.extend([True] * (len(detections.phrases) - idx - 1))
            break
        confirmed = answer.lower().startswith("yes")
        print(
            f"[VQA] '{phrase}' conf={float(detections.logits[idx]):.3f} "
            f"Q='{question}' A='{answer}' -> {'keep' if confirmed else 'drop'}",
            flush=True,
        )
        keep.append(confirmed)

    detections._filter(torch.tensor(keep, dtype=torch.bool))
