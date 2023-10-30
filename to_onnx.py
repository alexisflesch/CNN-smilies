"""
Script to convert the pytorch model to onnx for porting it to javascript.
"""

import torch

from neural_networks import CNN


def main():
    pytorch_model = CNN()
    pytorch_model.load_state_dict(torch.load('smiley_model.pth'))
    pytorch_model.eval()
    dummy_input = torch.zeros(1, 1, 32, 32)
    torch.onnx.export(pytorch_model,
                      dummy_input,
                      'smiley.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      verbose=True)


if __name__ == '__main__':
    main()
