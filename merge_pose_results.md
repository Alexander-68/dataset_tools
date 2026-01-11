# Merge YOLO Pose Results

This script merges YOLO pose estimation labels from two sources:
1.  **`labels-x` (Base):** The primary annotation source (e.g., full body pose).
2.  **`labels-face` (Face):** A secondary source specialized in faces (e.g., from a face-specific model or run).

The goal is to refine the face keypoints (Nose, Eyes, Ears) in the base annotations using the potentially more accurate or specific face annotations.

## Logic

1.  **Matching (Strict):**
    *   Iterates through each person (Class ID 0) in `labels-x`.
    *   Attempts to find a matching person in `labels-face` for the same image.
    *   **Strategy 1 (IoU):** Matches bounding boxes using Intersection over Union (Threshold: 0.45). This is the primary and most reliable method.
    *   **Strategy 2 (Face Center):** If IoU match fails, it attempts a strict match based on the proximity of face centers (Threshold: 0.05). Both objects must have visible face points for this to apply.
    *   **Unmatched:** If no reliable match is found, the person's face points are **not processed** and are kept exactly as they appear in `labels-x`.

2.  **Merging Keypoints (For Matched Persons):**
    *   Focuses on face keypoints: **Nose (0), Left Eye (1), Right Eye (2), Left Ear (3), Right Ear (4)**.
    *   **Absent in Face Source:** If a face point is missing in the matched `labels-face` entry, it is **removed** from the final annotation.
    *   **Present only in Face Source:** If the point exists in `labels-face` but not `labels-x`, it is **added**.
    *   **Present in Both:** The (x, y) coordinates are **averaged**.

3.  **Output:**
    *   Preserves the bounding box and body keypoints (Shoulders, Hips, etc.) from `labels-x`.
    *   Writes the refined annotations to the specified output folder.

## Usage

Run the script from the directory containing your label folders (CWD):

```bash
python merge_pose_results.py --labels-x "labels-x" --labels-face "labels-face" --output "labels"
```

### Arguments

*   `--labels-x`: Path to the folder containing the base body annotations (relative to CWD). Default: `labels-x`.
*   `--labels-face`: Path to the folder containing the face annotations (relative to CWD). Default: `labels-face`.
*   `--output`: Path to the output folder where merged labels will be saved (relative to CWD). Default: `labels`.

## Paths

*   Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
*   Script directory: contains the script and its `.md` description.
