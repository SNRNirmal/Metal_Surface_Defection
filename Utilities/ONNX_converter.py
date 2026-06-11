#!/usr/bin/env python
# coding: utf-8

import segmentation_models_pytorch as smp
import torch
from collections import OrderedDict
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')

    parser.add_argument('Model', type=str)
    parser.add_argument('Encoder', type=str)
    parser.add_argument('Encoder_Weights', type=str)
    parser.add_argument('Classes', type=int)
    parser.add_argument('Model_Path', type=str)
    parser.add_argument('Output_Path', type=str)

    args = parser.parse_args()

    model_name = args.Model
    ENCODER = args.Encoder
    ENCODER_WEIGHTS = None if args.Encoder_Weights.lower() == "none" else args.Encoder_Weights
    CLASSES = args.Classes
    ckpt_path = args.Model_Path
    output_path = args.Output_Path

    print(f"Model name: {model_name}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Output ONNX path: {output_path}")

    #  Check file exists
    if not os.path.exists(ckpt_path):
        print(f" Checkpoint not found: {ckpt_path}")
        return

    #  Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ACTIVATION = None


    if model_name == 'Unet':
        model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                         in_channels=3, classes=CLASSES, activation=ACTIVATION)

    elif model_name == 'FPN':
        model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=3, classes=CLASSES, activation=ACTIVATION)

    else:
        print(" Unknown model type")
        return

    # ---------------- Load Checkpoint ----------------
    state = torch.load(ckpt_path, map_location='cpu',weights_only=False)

    #  Handle multiple formats
    if isinstance(state, dict) and 'state_dict' in state:
        raw_state_dict = state['state_dict']
    elif isinstance(state, dict):
        raw_state_dict = state
    else:
        print(" Unsupported checkpoint format")
        return

    #  Remove DataParallel prefix
    new_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print(f" Missing keys: {len(missing)}")
    if unexpected:
        print(f" Unexpected keys: {len(unexpected)}")

    model = model.cpu().eval()

    input_var = torch.randn(1, 3, 128, 128)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                input_var,
                output_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        print(" ONNX model exported successfully!")

    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == "__main__":
    main()