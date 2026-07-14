from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from numpy import dot
from numpy.linalg import norm
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_pretrained = AutoModel.from_pretrained("google/vit-base-patch16-224")
vit_finetuned = AutoModel.from_pretrained("rfrade/near_duplicate_retrieval_vit")

img_paths = "example_imgs"
messi1 = Image.open(f"{img_paths}/messi_1.png").convert("RGB")
messi2 = Image.open(f"{img_paths}/messi_2.png").convert("RGB")

twiter_1 = Image.open(f"{img_paths}/twiter_1.png").convert("RGB")
twiter_2 = Image.open(f"{img_paths}/twiter_2.png").convert("RGB")

def similarity_images(model, img1, img2):

    preprocessed_1 = processor(img1, return_tensors="pt")
    preprocessed_2 = processor(img2, return_tensors="pt")

    with torch.no_grad():
        emb1 = model(**preprocessed_1).pooler_output.cpu().numpy()[0]
        emb2 = model(**preprocessed_2).pooler_output.cpu().numpy()[0]

    return cos_sim(emb1, emb2)

## INCREASES THE SIMILARITY OF MODIFIED VERSIONS OF THE SAME IMAGE
print("Similarity of modified versions of the same image")
print(f"pretrained: {similarity_images(vit_pretrained, messi1, messi2)}")
print(f"fine-tuned: {similarity_images(vit_finetuned, messi1, messi2)}", "\n\n")
#pretrained: 0.52
#fine-tuned: 0.57

## DECREASES THE SIMILARITY OF DIFFERENT IMAGES WITH SAME LAYOUT
print(f"Similarity of different images with the same layout")
print(f"pretrained: {similarity_images(vit_pretrained, twiter_1, twiter_2)}")
print(f"fine-tuned: {similarity_images(vit_finetuned, twiter_1, twiter_2)}")
#pretrained: 0.49
#fine-tuned: 0.44
