"""Grounding components of the GS metric: OWLv2 detector and IoU."""

import torch
from PIL import Image
from torchvision.ops import box_iou

from . import constants as C


class OWLv2Grounder:
    def __init__(self, device: str = "cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(C.OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(C.OWL_MODEL).to(device).eval()

    @torch.no_grad()
    def best_box(self, rgb, phrase: str):
        """Top-1 box [x1, y1, x2, y2] for `phrase` above the score threshold, or None."""
        image = Image.fromarray(rgb).convert("RGB")
        inputs = self.processor(text=[[phrase]], images=image, return_tensors="pt",
                                truncation=True).to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        result = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=C.OWL_SCORE_THRESHOLD)[0]
        if len(result["scores"]) == 0:
            return None
        best = result["scores"].argmax()
        return [round(v, 2) for v in result["boxes"][best].cpu().tolist()]


def iou(pred_box, gt_box) -> float:
    if pred_box is None:
        return 0.0
    return box_iou(torch.tensor([pred_box], dtype=torch.float),
                   torch.tensor([gt_box], dtype=torch.float)).item()
