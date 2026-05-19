"""
Recognize a handwritten digit from a photo using a trained MNIST model.
Usage:
  py recog_num.py photo_8.jpg
  py recog_num.py photo_9.jpg --weights weights.pth
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from mnist_from_photo import MnistSample, jpeg_to_mnist, save_preview

class MnistModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.layer1 = nn.Linear(28 * 28, 256)
		self.relu1 = nn.ReLU()
		self.layer2 = nn.Linear(256, 128)
		self.relu2 = nn.ReLU()
		self.layer3 = nn.Linear(128, 10)

	def forward(self, x: Tensor) -> Tensor:
		x = self.relu1(self.layer1(x))
		x = self.relu2(self.layer2(x))
		return self.layer3(x)

def show_preview(sample: MnistSample) -> None:
	"""Display the 28x28 MNIST-style image (same view as mnist.data in the notebook)."""
	plt.imshow(sample.image_uint8, cmap="gray")
	plt.axis("off")
	plt.show()

def predict(sample: MnistSample, weights_path: Path) -> tuple[int, Tensor]:
	model = MnistModel()
	state = torch.load(weights_path, map_location="cpu", weights_only=True)
	model.load_state_dict(state)
	model.eval()
	with torch.no_grad():
		logits = model(sample.as_model_input(batch=True))
		probs = torch.softmax(logits, dim=1)
		pred = int(probs.argmax(dim=1).item())
	return pred, probs.squeeze(0)


def main() -> None:
	p = argparse.ArgumentParser(description="Photo -> MNIST -> digit prediction")
	p.add_argument("image", type=Path)
	p.add_argument("--weights", type=Path, default=Path("weights.pth"))
	p.add_argument("--preview", type=Path, default=None)
	p.add_argument("--no-predict", action="store_true", help="Only convert, do not run model")
	args = p.parse_args()

	sample = jpeg_to_mnist(args.image)
	show_preview(sample)

	preview = args.preview or args.image.with_name(f"{args.image.stem}_mnist.png")
	save_preview(sample, preview)
	#print(f"MNIST preview saved: {preview}")
	#print(f"tensor_flat: shape={tuple(sample.tensor_flat.shape)}, min={sample.tensor_flat.min():.3f}, max={sample.tensor_flat.max():.3f}")

	if args.no_predict:
		return

	if not args.weights.is_file():
		print(f"Weights not found ({args.weights}); skipping prediction. Train/save weights first.")
		return

	digit, probs = predict(sample, args.weights)
	conf = float(probs[digit].item())
	print(f"Prediction: {digit}  (confidence {conf:.1%})")
	top3 = probs.topk(3)
	print("Top 3:", ", ".join(f"{int(i)}={p:.1%}" for i, p in zip(top3.indices, top3.values)))
	#print("Check *_mnist.png — it should look like a small MNIST digit (thin, centered).")

if __name__ == "__main__":
	main()
