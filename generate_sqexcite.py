import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from typing import List, Dict, Optional, Union
import torch

import sys
sys.path.append(".")
sys.path.append("..")
from PIL import Image

from pipeline_attend_and_excite import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils.ptp_utils import AttentionStore


NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# stable = AttendAndExcitePipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
#                                                      torch_dtype=torch.float16).to(device)

stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
tokenizer = stable.tokenizer

def generate_images_for_method(prompt: str,
                               seeds: List[int],
                               indices_to_alter: Optional[List[int]] = None,
                               is_attend_and_excite: bool = True):
    token_indices = get_indices_to_alter(stable, prompt) if indices_to_alter is None else indices_to_alter
    images = []
    for seed in seeds:
        g = torch.Generator('cuda').manual_seed(seed)
        prompts = [prompt]
        controller = AttentionStore()
        run_standard_sd = False if is_attend_and_excite else True
        image = run_and_display(prompts=prompts,
                                controller=controller,
                                indices_to_alter=token_indices,
                                generator=g,
                                run_standard_sd=run_standard_sd)
        images.append(image.resize((256, 256)))

        vis_utils.show_cross_attention(attention_store=controller,
                                       prompt=prompt,
                                       tokenizer=tokenizer,
                                       res=16,
                                       from_where=("up", "down", "mid"),
                                       indices_to_alter=token_indices,
                                       orig_image=image)

    return images

def run_and_display(prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    generator: torch.Generator,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    thresholds: Dict[int, float] = {10: 0.5, 20: 0.8},
                    max_iter_to_alter: int = 25,
                    display_output: bool = False):
    config = RunConfig(prompt=prompts[0],
                       run_standard_sd=run_standard_sd,
                       scale_factor=scale_factor,
                       thresholds=thresholds,
                       max_iter_to_alter=max_iter_to_alter)
    image = run_on_prompt(model=stable,
                          prompt=prompts,
                          controller=controller,
                          token_indices=indices_to_alter,
                          seed=generator,
                          config=config)
    return image

def main():

    prompt = "a cat and a frog"

    imgs = generate_images_for_method(
        prompt=prompt,  # "a cat and a frog",
        seeds=[6141, 9031, 969, 1910],
        indices_to_alter = [1,5],
        is_attend_and_excite=False
    )


    imgs = generate_images_for_method(
        prompt=prompt,  # "a cat and a frog",
        seeds=[6141, 9031, 969, 1910],
        indices_to_alter=[1, 5],
        is_attend_and_excite=True
    )

    vis_utils.show_cross_attention(attention_store=controller,
                                   prompt=prompt,
                                   tokenizer=tokenizer,
                                   res=16,
                                   from_where=("up", "down", "mid"),
                                   indices_to_alter=token_indices,
                                   orig_image=image)


if __name__ == '__main__':
    main()