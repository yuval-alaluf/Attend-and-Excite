import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch

from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention


def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None,
                         vis_image_size=512,
                         auto_scale_heatmap=True,
                         heatmap_scale=100.0):
    """
        Args:
            vis_image_size: width/height of each image, in the final image grid
            auto_scale_heatmap: if True, each image's attention heatmap gets normalized independently,
                to make it easier to see the relative attention within that image. 
                If False, all images' heatmaps get scaled by `heatmap_scale`
            heatmap_scale: amount by which to scale the heatmaps.
    """
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image, vis_image_size, auto_scale_heatmap, heatmap_scale)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((vis_image_size, vis_image_size)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))


def show_image_relevance(image_relevance, image: Image.Image, vis_image_size=512, auto_scale_heatmap=True,
                         heatmap_scale=100.0):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        if auto_scale_heatmap:
            cam = cam / np.max(cam)
        else:
            cam = cam / 2.0 # normalize by constant value
        return cam

    image = image.resize((vis_image_size, vis_image_size))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=vis_image_size, mode='bilinear') # nearest would be better
    if auto_scale_heatmap:
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    else:
        image_relevance *= heatmap_scale
    image_relevance = image_relevance.reshape(vis_image_size, vis_image_size)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
