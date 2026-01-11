# extract_tfrecord_images.py

Extract images from a TFRecord file and write them as JPEG files (quality 95 by
default) into an output folder (default `images/`).

## Usage

```bash
python extract_tfrecord_images.py path\to\data.tfrec
```

```bash
python extract_tfrecord_images.py data.tfrecord --output images --quality 95
```

## Arguments

- `tfrecord`: Path to the TFRecord file to read.
- `--output`: Output directory for extracted images. Default: `./images`.
- `--quality`: JPEG quality (1-100). Default: 95.
- `--image-key`: Optional override for the image feature key (e.g. `image/encoded`).
- `--filename-key`: Optional override for the filename feature key
  (e.g. `image/filename`).

## Notes

- The script reads records with `tensorflow`. Install it if missing:
  `pip install tensorflow`.
- If no `--image-key` is provided, it tries common keys like `image/encoded` and
  falls back to any bytes field that decodes as an image.
- Output filenames use the TFRecord filename field when available; otherwise
  they are numbered like `image_000001.jpg`.
