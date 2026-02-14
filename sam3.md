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
- Prints startup parameters, progress with ETA, and final stats.

## Parameters

- `--input-image` (required): image file to process. Relative paths are from CWD.
- `--targets` (required): one or more text targets for prediction.
- `--model` (optional): SAM3 model file path. Default: `sam3.pt` in Script Directory.
- `--conf` (optional): confidence threshold. Default: `0.25`.
- `--half` (optional flag): enable FP16 inference.

## YOLO output format

Each predicted box is written as one line:

`class_id x_center y_center width height`

All coordinates are normalized to `[0, 1]` (`xywhn`).

## Usage examples

```bash
python sam3.py --input-image images/person1.jpg --targets eye nose ear hand foot
```

```bash
python sam3.py --input-image image.jpg --targets person --conf 0.3 --half
```

```bash
python sam3.py --input-image images/sample.png --targets bottle cap label --model sam3.pt
```

## Notes

- If no boxes are predicted, an empty `.txt` file is still created.
- Segmentation preview is always saved as `.png` with the same input stem.
- Output `.txt` is written next to the input image.
- Preview display is segmentation-only for visual clarity.
- Ensure `ultralytics` and the SAM3 model file are available.
