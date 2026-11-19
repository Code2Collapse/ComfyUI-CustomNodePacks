# BBoxCrop (legacy): core Crop Image (ImageCropV2) + Crop Mask

**Status:** still registered and marked DEPRECATED. It is not migrated automatically: the replacement is two core
nodes, and the old BBOX type is not core's BOUNDING_BOX. It logs one warning per run that describes the replacement.

| Old input / output | Core replacement |
|---|---|
| image + bbox `[x, y, w, h]` -> cropped_image | Crop Image, crop_region `{x, y, width: w, height: h}` |
| mask -> cropped_mask | Crop Mask with the same x, y, width, height |
| no mask -> cropped_mask = ones | none (use Create Solid Mask of the crop size) |
| bbox output (clamped box) | none |

**Parity:** bit-exact (`torch.equal`) for image and mask with box `[7, 5, 20, 16]` on a 2x40x50 batch. It is also
bit-exact for a box past the right and bottom edges (`[40, 30, 30, 30]`), where both slice to the same 10x10 region.
The core `execute` bodies are from the installed core 0.39.0 file.

Differences:
- The old node resized a mask of another size to the image size before cropping. Crop Mask does not.
- An empty box became a 1x1 crop in the old node.
- Core's legacy `ImageCrop` (marked DEPRECATED in core 0.39) behaves the same as ImageCropV2 here.

**Owner decision (not taken):** keep it, or remove it. Removal needs a two-node expansion and a BBOX ->
BOUNDING_BOX conversion.
