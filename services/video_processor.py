import gc
import re
import open_clip
import torch
from config import MODEL_NAME, PRETRAINED_SOURCE


class VideoProcessor:

  def __init__(self):
    self.device = "cpu"
    print("Cargando modelo OpenCLIP para CPU...")

    # Cargar modelo y transformaciones
    self.model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_SOURCE, device=self.device
    )
    self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # 1. Modo evaluación estricto
    self.model.eval()

    # 2. Desactivar gradientes en todos los parámetros por seguridad de RAM
    for param in self.model.parameters():
      param.requires_grad = False

    # 3. Forzar limpieza de basura en RAM tras la carga inicial pesada
    gc.collect()
    print("Modelo cargado y memoria optimizada con éxito.")

  def encode_text_query(self, text: str) -> list[float]:
    with torch.no_grad():
      tokens = self.tokenizer([text]).to(self.device)
      features = self.model.encode_text(tokens)
      features /= features.norm(dim=-1, keepdim=True)
      return features.cpu().numpy()[0].tolist()