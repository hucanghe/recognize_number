"""
Convert a JPEG (or other image) of a handwritten digit into MNIST-compatible data.

Output matches this project's training pipeline:
  - uint8 array (28, 28), black background, bright digit (like torchvision MNIST .data)
  - float tensor (784,) in [0, 1] from torchvision.transforms.functional.to_tensor
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional


MNIST_SIZE = 28
# MNIST digits occupy roughly a 20x20 box inside the 28x28 frame.
MNIST_DIGIT_BOX = 20


@dataclass(frozen=True)
class MnistSample:
	"""One digit in MNIST-style layout."""

	# Like mnist.data[i]: uint8, shape (28, 28), 0 = background
	image_uint8: np.ndarray
	# Like img_preprocess(mnist[i][0]): float, shape (784,), values in [0, 1]
	tensor_flat: Tensor

	@property
	def image_2d(self) -> np.ndarray:
		return self.image_uint8

	@property
	def tensor_chw(self) -> Tensor:
		return self.tensor_flat.view(1, MNIST_SIZE, MNIST_SIZE)

	def as_model_input(self, batch: bool = False) -> Tensor:
		"""Same layout as mnist_model(images) in training (N, 784) or (784,)."""
		if batch:
			return self.tensor_flat.unsqueeze(0)
		return self.tensor_flat


def _load_image(path: str | Path) -> Image.Image:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(path)
	img = Image.open(path)
	img.load()
	return img.convert("L")


def _otsu_threshold(gray: np.ndarray) -> int:
	hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
	total = gray.size
	sum_total = np.dot(np.arange(256), hist)
	sum_b, w_b, max_var, threshold = 0.0, 0.0, 0.0, 128
	for t in range(256):
		w_b += hist[t]
		if w_b == 0:
			continue
		w_f = total - w_b
		if w_f == 0:
			break
		sum_b += t * hist[t]
		m_b = sum_b / w_b
		m_f = (sum_total - sum_b) / w_f
		var_between = w_b * w_f * (m_b - m_f) ** 2
		if var_between > max_var:
			max_var = var_between
			threshold = t
	return int(threshold)


def _paper_is_bright(gray: np.ndarray) -> bool:
	h, w = gray.shape
	margin = max(2, min(h, w) // 10)
	corners = np.concatenate([
		gray[:margin, :margin].ravel(),
		gray[:margin, -margin:].ravel(),
		gray[-margin:, :margin].ravel(),
		gray[-margin:, -margin:].ravel(),
	])
	return float(corners.mean()) > 127.0


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
	"""Mask of digit pixels on the original photo (before MNIST polarity)."""
	th = _otsu_threshold(gray)
	if _paper_is_bright(gray):
		return gray < th
	return gray > th


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
	if not mask.any():
		h, w = mask.shape
		return 0, 0, h, w
	rows = np.where(mask.any(axis=1))[0]
	cols = np.where(mask.any(axis=0))[0]
	return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def _to_mnist_polarity(gray: np.ndarray) -> np.ndarray:
	"""Black background (0), bright digit (255)."""
	if _paper_is_bright(gray):
		return 255 - gray
	return gray


def _crop_digit_square(gray: np.ndarray, padding_ratio: float = 0.2) -> np.ndarray:
	"""Crop around ink on the original photo, then caller converts polarity."""
	mask = _foreground_mask(gray)
	top, left, bottom, right = _bounding_box(mask)
	crop = gray[top:bottom, left:right]
	ch, cw = crop.shape
	if ch == 0 or cw == 0:
		return gray
	pad = int(padding_ratio * max(ch, cw))
	side = max(ch, cw) + 2 * pad
	square = np.full((side, side), 255 if _paper_is_bright(gray) else 0, dtype=np.uint8)
	y0 = (side - ch) // 2
	x0 = (side - cw) // 2
	square[y0 : y0 + ch, x0 : x0 + cw] = crop
	return square


def _center_of_mass(mask: np.ndarray) -> tuple[float, float]:
	ys, xs = np.nonzero(mask)
	if len(xs) == 0:
		return mask.shape[0] / 2, mask.shape[1] / 2
	return float(ys.mean()), float(xs.mean())


def _fit_to_mnist_canvas(digit: np.ndarray) -> np.ndarray:
	"""
	Resize digit to MNIST_DIGIT_BOX and center in 28x28 (LeCun-style layout).
	digit: uint8, black background, bright strokes.
	"""
	mask = digit > 30
	top, left, bottom, right = _bounding_box(mask)
	crop = digit[top:bottom, left:right]
	ch, cw = crop.shape
	if ch == 0 or cw == 0:
		return np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)

	scale = MNIST_DIGIT_BOX / max(ch, cw)
	new_h = max(1, int(round(ch * scale)))
	new_w = max(1, int(round(cw * scale)))
	resized = np.array(
		Image.fromarray(crop, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS),
		dtype=np.uint8,
	)

	canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
	r_mask = resized > 30
	cy, cx = _center_of_mass(r_mask)
	target_cy, target_cx = (MNIST_SIZE - 1) / 2.0, (MNIST_SIZE - 1) / 2.0
	y0 = int(round(target_cy - cy))
	x0 = int(round(target_cx - cx))
	for y in range(new_h):
		for x in range(new_w):
			yy, xx = y0 + y, x0 + x
			if 0 <= yy < MNIST_SIZE and 0 <= xx < MNIST_SIZE:
				canvas[yy, xx] = max(canvas[yy, xx], resized[y, x])
	return canvas


def photo_to_mnist_pil(
	image: Image.Image,
	*,
	invert: bool | None = None,
	padding_ratio: float = 0.2,
) -> Image.Image:
	"""
	Resize and normalize a grayscale PIL image to 28x28 MNIST-style (L mode).
	"""
	gray = np.array(image.convert("L"), dtype=np.uint8)

	# Crop on original photo (bbox must use ink-on-paper, not MNIST polarity).
	square = _crop_digit_square(gray, padding_ratio=padding_ratio)
	digit = _to_mnist_polarity(square)
	if invert is True:
		digit = 255 - digit
	elif invert is False:
		pass
	# invert is None: _to_mnist_polarity already handled paper brightness

	canvas = _fit_to_mnist_canvas(digit)
	return Image.fromarray(canvas, mode="L")


def pil_to_mnist_sample(pil_28: Image.Image) -> MnistSample:
	image_uint8 = np.array(pil_28, dtype=np.uint8)
	tensor = functional.to_tensor(pil_28).view(MNIST_SIZE * MNIST_SIZE)
	return MnistSample(image_uint8=image_uint8, tensor_flat=tensor)


def jpeg_to_mnist(
	path: str | Path,
	*,
	invert: bool | None = None,
	padding_ratio: float = 0.2,
) -> MnistSample:
	"""Load a JPEG/PNG and return MNIST-compatible data."""
	pil = photo_to_mnist_pil(_load_image(path), invert=invert, padding_ratio=padding_ratio)
	return pil_to_mnist_sample(pil)


def save_preview(sample: MnistSample, path: str | Path) -> None:
	"""Save the 28x28 grayscale preview as PNG."""
	Image.fromarray(sample.image_uint8, mode="L").save(path)


def save_tensor(sample: MnistSample, path: str | Path) -> None:
	"""Save flat tensor for later model inference."""
	torch.save(sample.tensor_flat, path)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Convert a handwritten-digit photo to MNIST-style tensors.",
	)
	p.add_argument("image", type=Path, help="JPEG/PNG path")
	p.add_argument(
		"--preview",
		type=Path,
		default=None,
		help="Write 28x28 PNG preview (default: <stem>_mnist.png)",
	)
	p.add_argument(
		"--tensor",
		type=Path,
		default=None,
		help="Write .pt file with flat (784,) tensor (default: <stem>_mnist.pt)",
	)
	p.add_argument(
		"--invert",
		action="store_true",
		help="Force invert after crop",
	)
	p.add_argument(
		"--no-invert",
		action="store_true",
		help="Force no invert after crop",
	)
	p.add_argument(
		"--padding",
		type=float,
		default=0.2,
		help="Padding around digit crop as fraction of box size (default: 0.2)",
	)
	return p


def main(argv: list[str] | None = None) -> MnistSample:
	args = _build_parser().parse_args(argv)
	invert: bool | None = True if args.invert else (False if args.no_invert else None)
	sample = jpeg_to_mnist(args.image, invert=invert, padding_ratio=args.padding)
	stem = args.image.stem
	preview_path = args.preview or args.image.with_name(f"{stem}_mnist.png")
	tensor_path = args.tensor or args.image.with_name(f"{stem}_mnist.pt")
	save_preview(sample, preview_path)
	save_tensor(sample, tensor_path)
	print(f"uint8 shape: {sample.image_uint8.shape}, range [{sample.image_uint8.min()}, {sample.image_uint8.max()}]")
	print(f"tensor shape: {tuple(sample.tensor_flat.shape)}, dtype {sample.tensor_flat.dtype}")
	print(f"preview: {preview_path}")
	print(f"tensor:  {tensor_path}")
	return sample


if __name__ == "__main__":
	main()
