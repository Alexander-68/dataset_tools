# sam3.py

Run SAM3 text-prompted prediction on a single image, save a segmentation preview
as PNG, and save predicted bounding boxes in YOLO `.txt` format.

The output files use the same input image stem in the same folder:
- segmentation preview: `.png`
- YOLO labels: `.txt`

## What it does

- Resolves **input image** relative to Current Working Directory (CWD).
- Resolves **model path** relative to Script Directory by default.
- Runs `SAM3SemanticPredictor` with your text targets.
- Saves segmentation-only preview image as `<input_image_stem>.png`.
- Shows prediction preview with segmentation masks only (no boxes, no class labels).
- Writes YOLO detection labels to `<input_image_stem>.txt`.
- Supports selecting bbox source: model boxes or mask-derived boxes.
- Always applies NMS for same-class intersecting boxes before writing labels.
- Prints startup parameters, progress with ETA, and final stats including total runtime.

## Parameters

- `--input-image` (required): image file to process. Relative paths are from CWD.
- `--targets` (required): one or more text targets for prediction.
- `--model` (optional): SAM3 model file path. Default: `sam3.pt` in Script Directory.
- `--conf` (optional): confidence threshold. Default: `0.25`.
- `--half` (optional flag): enable FP16 inference.
- `--imgsz` (optional): requested inference image size. Default: `640`.
- `--stride-align` (optional): stride multiple used to auto-align `--imgsz`. Default: `14`.
- `--bbox-source` (optional): source for YOLO boxes.
  - `model` (default): use `results[0].boxes` from SAM3 output.
  - `mask`: compute box extents directly from segmentation masks.

## YOLO output format

Each predicted box is written as one line:

`class_id x_center y_center width height`

All coordinates are normalized to `[0, 1]` (`xywhn` semantics).

## Usage examples

```bash
python sam3.py --input-image images/person1.jpg --targets eye nose ear hand foot
```

```bash
python sam3.py --input-image image.jpg --targets person --conf 0.3 --half --bbox-source mask
```

```bash
python sam3.py --input-image images/sample.png --targets bottle cap label --model sam3.pt --imgsz 640 --bbox-source model
```

## Notes

- If no boxes are predicted, an empty `.txt` file is still created.
- Segmentation preview is always saved as `.png` with the same input stem.
- Output `.txt` is written next to the input image.
- Preview display is segmentation-only for visual clarity.
- `--imgsz` is aligned to `--stride-align` before inference (example: `640 -> 644` for stride `14`) to avoid stride warnings.
- NMS is always applied per class and suppresses any intersecting box (IoU > 0).
- `--bbox-source mask` can better align bboxes with displayed segmentation overlays.
- Ensure `ultralytics` and the SAM3 model file are available.

