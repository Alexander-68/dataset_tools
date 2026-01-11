import argparse
from pathlib import Path

from tqdm import tqdm


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def process_label_file(
    file_path: Path, output_dir: Path, threshold: float = 0.4
) -> None:
    """
    Reads a YOLO pose annotation file, corrects keypoints for class 0,
    and writes the result to the output directory.
    """
    output_path = output_dir / file_path.name

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    corrected_lines = []
    for line in lines:
        parts = line.strip().split()
        
        # Skip empty lines
        if not parts:
            continue

        class_id = int(parts[0])

        # Process only if class_id is 0 (person)
        if class_id == 0:
            # YOLO pose format: class cx cy w h kpt1_x kpt1_y kpt1_v ...
            # Keypoints start at index 5
            # Each keypoint has 3 values: x, y, visibility
            
            # Check if we have keypoints (length > 5)
            if len(parts) > 5:
                keypoints = parts[5:]
                num_kpts = len(keypoints) // 3
                
                new_keypoints = []
                for i in range(num_kpts):
                    idx = i * 3
                    x = float(keypoints[idx])
                    y = float(keypoints[idx+1])
                    v_val = float(keypoints[idx+2])
                    
                    # Logic:
                    # 1. If visibility is a probability (0.0 to 1.0):
                    #    >= threshold -> 2, < threshold -> 0
                    # 2. If visibility is already > 1 (e.g., 2), keep it as is (integer)
                    
                    if 0.0 <= v_val <= 1.0:
                        v = 2 if v_val >= threshold else 0
                    else:
                        v = int(v_val)
                    
                    # Reset X and Y if visibility is 0
                    if v == 0:
                        x = 0.0
                        y = 0.0
                    
                    # Append corrected values
                    x_str = f"{x:g}" if x != 0 else "0.0"
                    y_str = f"{y:g}" if y != 0 else "0.0"
                    
                    new_keypoints.extend([x_str, y_str, str(v)])
                
                # Reconstruct the line
                # parts[:5] are class cx cy w h
                new_line_parts = parts[:5] + new_keypoints
                corrected_lines.append(" ".join(new_line_parts) + "\n")
            else:
                # No keypoints, just copy
                corrected_lines.append(line)
        else:
            # Not a person, just copy
            corrected_lines.append(line)

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct YOLO11-pose keypoints."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="labels",
        help="Folder containing label files (default: 'labels')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Visibility probability threshold (default: 0.4)",
    )

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    input_dir = resolve_path(Path(args.folder), cwd)
    threshold = args.threshold

    print(
        "Correcting YOLO11-pose keypoints.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- Labels dir: {input_dir}\n"
        f"- Threshold: {threshold}"
    )

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return 1

    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} label files in '{input_dir}'.")
    print(f"Processing files in-place with threshold {threshold}...")

    for txt_file in tqdm(txt_files, desc="Correcting", unit="file"):
        process_label_file(txt_file, input_dir, threshold)

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
