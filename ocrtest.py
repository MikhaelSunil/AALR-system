import torch
import easyocr

print("Torch CUDA available:", torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR is ready.")

