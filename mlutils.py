from torchvision import transforms
from torch.nn import functional as F
import torch
from transformers import ViTModel
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import os
from PIL import Image
from collections import defaultdict
import numpy as np


class MLProcessor:
    weights_name = 'pytorch_lora_weights.safetensors'

    def __init__(self, config) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            config['diffusion'], torch_dtype=torch.float16, safety_checker=None
        ).to(self.device)
        self.dino = ViTModel.from_pretrained('facebook/dino-vits16').to(self.device)

        # DINO Transforms
        self.T = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.reference_embeds = {}
        self.keyword2adapter = {}
        self.adapter2token = {}
        self.adapter2path = {}
        for adapter_name, values in config['adapters'].items():
            self.adapter2path[adapter_name] = values['path2adapter']
            embeds = []
            for file in os.listdir(values['path2references']):
                if '.jpg' in file:
                    embeds.append(self.get_dino_image_feats_norm(
                        Image.open(os.path.join(values['path2references'], file)))
                    )
            assert(len(embeds) > 0)
            self.reference_embeds[adapter_name] = embeds
            self.keyword2adapter[values['trigger']] = adapter_name
            self.adapter2token[adapter_name] = values['real_token']
        self.config = config

    def get_dino_image_feats_norm(self, image):
        with torch.no_grad():
            outputs = self.dino(self.T(image).unsqueeze(0).to(self.device))
            last_hidden_states = outputs.last_hidden_state # ViT backbone features
            embed = last_hidden_states[0, 0]
            embed /= torch.norm(embed)
            return embed

    def get_dino_score(self, ref_embeds, image):
        image_embed = self.get_dino_image_feats_norm(image)
        similarity = 0
        for embed in ref_embeds:
            similarity += (embed * image_embed).sum().item()
        return similarity / len(ref_embeds)

    def generate(self, init_image, mask_image, prompt, negative_prompt=None):
        prompt = ' ' + prompt + ' '
        if negative_prompt is not None:
            prompt = ' ' + negative_prompt + ' '

        adapter_name = None
        triggered_keyword = None
        for keyword in self.keyword2adapter:
            if keyword in prompt or (negative_prompt is not None and keyword in negative_prompt):
                if adapter_name is not None:
                    raise RuntimeError("You shouldn't use 2 concepts in one prompt. " 
                                       "Try to use them sequentially")
                adapter_name = self.keyword2adapter[keyword]      
                triggered_keyword = keyword
        if adapter_name is not None:
            
            self.pipeline.load_lora_weights(self.adapter2path[adapter_name], 
                                            weight_name=self.weights_name, 
                                            adapter_name=adapter_name)
            print('adapter_name', adapter_name)
            prompt = prompt.replace(triggered_keyword, self.adapter2token[adapter_name])
            if negative_prompt is not None:
                negative_prompt = negative_prompt.replace(triggered_keyword, self.adapter2token[adapter_name])
            self.pipeline.set_adapters([adapter_name])
        else:
            self.pipeline.set_adapters([])
        
        images = self.pipeline(prompt=prompt, negative_prompt=negative_prompt,
                               image=init_image, mask_image=mask_image,
                               guidance_scale=self.config['guidance_scale'], 
                               num_images_per_prompt=self.config['batch_size'],
                               cross_attention_kwargs={"scale": 1.0}).images
        if adapter_name is None:
            return images[:self.config['topk']]
        scores = []
        for i in range(len(images)):
            score = self.get_dino_score(self.reference_embeds[adapter_name], images[i])
            scores.append((score, i))
        scores.sort(reverse=True)
        self.pipeline.unload_lora_weights()
        return [images[el[1]] for el in scores[:self.config['topk']]]

    def list_concepts(self):
        concepts = {}
        for key, value in self.config['adapters'].items():
            concepts[value['trigger'].strip()] = value['description']
        return concepts

if __name__ == "__main__":
    import json
    os.makedirs('generated/', exist_ok=True)
    processor = MLProcessor(json.load(open('config.json')))
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
    images = processor.generate(init_image, mask_image, "a dog</w> dog")
    for i, image in enumerate(images):
        image.save(f"generated/{i + 1}.jpg")