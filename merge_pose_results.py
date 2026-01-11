import argparse
import numpy as np
from pathlib import Path


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def render_progress(current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    filled = int(round((current / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)

def parse_yolo_kpts(line):
    """
    Parses a YOLO pose line.
    Format: class x_center y_center width height px1 py1 pv1 ... pxn pyn pvn
    """
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    bbox = parts[1:5] # xc, yc, w, h
    kpts = parts[5:]
    
    # Reshape kpts to (N, 3)
    num_kpts = len(kpts) // 3
    kpts = np.array(kpts).reshape(num_kpts, 3)
    
    return {
        'class_id': class_id,
        'bbox': bbox,
        'kpts': kpts,
        'raw': parts
    }

def format_yolo_kpts(class_id, bbox, kpts):
    """
    Formats data back to YOLO line string.
    """
    xc, yc, w, h = bbox
    line_parts = [str(class_id), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
    
    for kp in kpts:
        line_parts.append(f"{kp[0]:.6f}")
        line_parts.append(f"{kp[1]:.6f}")
        line_parts.append(f"{kp[2]:.6f}") # Visibility should essentially be integer-like usually, but float is fine
        
    return " ".join(line_parts)

def compute_iou(bbox1, bbox2):
    """
    Computes IoU between two YOLO bboxes (xc, yc, w, h).
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to xmin, ymin, xmax, ymax
    b1_x1, b1_y1 = x1 - w1/2, y1 - h1/2
    b1_x2, b1_y2 = x1 + w1/2, y1 + h1/2
    
    b2_x1, b2_y1 = x2 - w2/2, y2 - h2/2
    b2_x2, b2_y2 = x2 + w2/2, y2 + h2/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    inter_area = inter_w * inter_h
    b1_area = w1 * h1
    b2_area = w2 * h2
    
    union_area = b1_area + b2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def get_face_center(kpts):
    """
    Calculates the centroid of visible face keypoints (indices 0-4).
    Returns None if no face keypoints are visible.
    """
    face_indices = [0, 1, 2, 3, 4]
    visible_pts = []
    for i in face_indices:
        if i < len(kpts) and kpts[i][2] > 0: # Assuming v > 0 means present
            visible_pts.append(kpts[i][:2])
            
    if not visible_pts:
        return None
        
    return np.mean(visible_pts, axis=0)

def merge_persons(p_x, p_face):
    """
    Merges person p_x (base) with p_face (face source).
    """
    # Create a copy of p_x keypoints to modify
    new_kpts = p_x['kpts'].copy()
    face_indices = [0, 1, 2, 3, 4]
    
    src_kpts = p_face['kpts']
    
    for i in face_indices:
        if i >= len(new_kpts) or i >= len(src_kpts):
            continue
            
        kp_x = new_kpts[i]
        kp_f = src_kpts[i]
        
        # Visibility check (assuming > 0 is present/visible)
        # Standard YOLO: 0=invisible, 1=visible (or occluded), 2=visible
        # Sometimes float confidence. We treat > 0 as present.
        
        present_x = kp_x[2] > 0
        present_f = kp_f[2] > 0
        
        if not present_f:
            # Rule: if absent in labels-face, remove from final
            new_kpts[i] = [0, 0, 0]
        elif present_f and not present_x:
            # Rule: if present in face but absent in x, add it
            new_kpts[i] = kp_f
        elif present_f and present_x:
            # Rule: if present in both, average x,y
            avg_x = (kp_x[0] + kp_f[0]) / 2
            avg_y = (kp_x[1] + kp_f[1]) / 2
            # For visibility, we can average or max. 
            # If both detect, likely reliable. Max or Avg? 
            # Prompt doesn't specify visibility merge, just x,y location.
            # I'll keep the visibility from labels-face as it's the "face specialist" source
            # or average them if they are confidences.
            # Let's average the third dim too for smoothness if it's confidence.
            avg_v = (kp_x[2] + kp_f[2]) / 2 
            new_kpts[i] = [avg_x, avg_y, avg_v]
            
    return new_kpts

def process_files(dir_x: Path, dir_face: Path, dir_out: Path) -> None:
    print(f"--- Starting Merge Pose Results ---")
    print(f"Parameters: labels-x='{dir_x}', labels-face='{dir_face}', output='{dir_out}'")
    print("Action: Merging face keypoints from 'labels-face' into 'labels-x' for matched persons.")

    dir_out.mkdir(parents=True, exist_ok=True)

    # Get all txt files in labels-x
    files_x = sorted(dir_x.glob("*.txt"))
    # Count files in labels-face
    files_face_count = len(list(dir_face.glob("*.txt")))

    print(f"Found {len(files_x)} files in Base folder '{dir_x}'")
    print(f"Found {files_face_count} files in Face folder '{dir_face}'")
    
    total_persons_x = 0
    faces_corrected = 0

    for idx, f_x_path in enumerate(files_x, start=1):
        filename = f_x_path.name
        f_face_path = dir_face / filename
        f_out_path = dir_out / filename

        # Read labels-x
        persons_x = []
        with f_x_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    persons_x.append(parse_yolo_kpts(line))

        # Read labels-face if exists
        persons_face = []
        if f_face_path.exists():
            with f_face_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        persons_face.append(parse_yolo_kpts(line))
        
        final_lines = []
        
        # Iterate over base persons (labels-x)
        for px in persons_x:
            if px['class_id'] != 0:
                # If not person, just copy? Prompt says "attempt to match 'person'". 
                # Assuming dataset is persons. If other classes exist, we might want to preserve them.
                # I'll preserve them as is.
                final_lines.append(format_yolo_kpts(px['class_id'], px['bbox'], px['kpts']))
                continue
            
            total_persons_x += 1
                
            # Find match in persons_face
            best_match = None
            best_iou = -1
            best_idx = -1
            
            # 1. Attempt IoU match (The "best way" to match bboxes)
            for idx, pf in enumerate(persons_face):
                if pf['class_id'] != 0: continue
                iou = compute_iou(px['bbox'], pf['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # Threshold for IoU (Strict to avoid 'havoc')
            IOU_THRESH = 0.45 
            
            if best_iou > IOU_THRESH:
                best_match = persons_face[best_idx]
            else:
                # 2. Attempt Face Keypoints Center Match (Strict Fallback)
                px_face_center = get_face_center(px['kpts'])
                
                min_dist = float('inf')
                best_dist_idx = -1
                
                if px_face_center is not None:
                    for idx, pf in enumerate(persons_face):
                        if pf['class_id'] != 0: continue
                        
                        pf_face_center = get_face_center(pf['kpts'])
                        if pf_face_center is None: continue # Cannot match center if pf has no face pts
                            
                        dist = np.linalg.norm(px_face_center - pf_face_center)
                        if dist < min_dist:
                            min_dist = dist
                            best_dist_idx = idx
                            
                    # Very strict distance threshold for normalized coordinates
                    DIST_THRESH = 0.05 
                    if min_dist < DIST_THRESH and best_dist_idx != -1:
                        best_match = persons_face[best_dist_idx]

            # Merge
            final_kpts = None
            if best_match:
                faces_corrected += 1
                # Process face points only when a match is found
                final_kpts = merge_persons(px, best_match)
            else:
                # "when an object cannot find a match - do not process it's face points"
                # This means keep the original keypoints from px as they are.
                final_kpts = px['kpts'].copy()
            
            final_lines.append(format_yolo_kpts(px['class_id'], px['bbox'], final_kpts))
            
        # Write output
        with f_out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(final_lines))
        render_progress(idx, len(files_x))

    if files_x:
        print()
    print(f"--- Processing Complete ---")
    print(f"Total annotations (files) processed: {len(files_x)}")
    print(f"Total persons found in Base: {total_persons_x}")
    print(f"Faces corrected (matched): {faces_corrected}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge YOLO pose labels.")
    parser.add_argument("--labels-x", default="labels-x", help="Folder with base body annotations")
    parser.add_argument("--labels-face", default="labels-face", help="Folder with face annotations")
    parser.add_argument("--output", default="labels", help="Output folder")

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    labels_x = resolve_path(Path(args.labels_x), cwd)
    labels_face = resolve_path(Path(args.labels_face), cwd)
    output_dir = resolve_path(Path(args.output), cwd)
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    process_files(labels_x, labels_face, output_dir)
