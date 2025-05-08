import torch
import sys

def inspect_pth(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        for key, val in ckpt.items():
            print(f"\nKey: {key}")
            if isinstance(val, dict):
                print(f"  Nested dict with {len(val)} entries.")
                for subkey, subval in val.items():
                    if torch.is_tensor(subval):
                        print(f"    {subkey}: Tensor, shape={tuple(subval.shape)}, dtype={subval.dtype}")
                    elif isinstance(subval, (int, float, str, list)):
                        print(f"    {subkey}: {type(subval).__name__}, value={subval}")
                    else:
                        print(f"    {subkey}: {type(subval).__name__}")
            elif torch.is_tensor(val):
                print(f"  Tensor, shape={tuple(val.shape)}, dtype={val.dtype}")
            else:
                print(f"  {type(val).__name__}")
    else:
        print("Loaded object is not a dictionary. It is of type:", type(ckpt).__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pth.py <path_to_pth_file>")
    else:
        inspect_pth(sys.argv[1])

