from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch 
import torch.distributed as dist
import torch
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
)

# This file is used to create diffusion models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_guided_diffusion_model(args_):
    model, diffusion = create_model_and_diffusion(
        **{k: args_.get(k,v) for k,v in model_and_diffusion_defaults().items()}
    )

    model.load_state_dict(torch.load(args_["model_path"]))
    model.eval()
    model = model.to(device)
    print("Loaded model from", args_["model_path"])
    return model, diffusion

def create_256_cond(clf=False):
    args ={
    "attention_resolutions": "32, 16, 8",
    "class_cond": True,
    "diffusion_steps": 1000,
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_new_attention_order": True,
    "use_fp16": False,
    "use_scale_shift_norm": True,
    "model_path": "256x256_diffusion.pt",
    "num_samples":10,
    "timestep_respacing": "250",    
    }
    model, diffusion = create_guided_diffusion_model(args)
    if clf:
        classifier = create_classifier(
            **{k: args.get(k,v) for k,v in classifier_defaults().items()}
        )

        classifier.load_state_dict(torch.load("256x256_classifier.pt"))
        classifier.eval()
        classifier = classifier.to(device)
        return model, diffusion, classifier
    return model, diffusion



def create_256_uncond():
    args ={
    "attention_resolutions": "32, 16, 8",
    "class_cond": False,
    "diffusion_steps": 1000,
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_new_attention_order": True,
    "use_fp16": False,
    "use_scale_shift_norm": True,
    "model_path": "256x256_diffusion_uncond.pt",
    "num_samples":10,
    "timestep_respacing": "250",    
    }
    return create_guided_diffusion_model(args)

def create_64_uncond():
    args ={
        "attention_resolutions": "32, 16, 8",
        "class_cond": True,
        "diffusion_steps": 1000,
        "dropout": 0.1,
        "image_size": 64,
        "learn_sigma": True,
        "noise_schedule": "cosine",
        "num_channels": 192,
        "num_head_channels": 64,
        "num_res_blocks": 3,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "use_fp16": False,
        "use_scale_shift_norm": True,
        "model_path": "64x64_diffusion.pt",
        "num_samples":10,
        "timestep_respacing": "250",
    }
    return create_guided_diffusion_model(args)