import torch
import torchaudio

from pathlib import Path
from argparse import ArgumentParser


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="F:/Study/Language Model/Voice Changer/hubert/weight/hubert_discrete_pretrained.pt",
    )
    args = parser.parse_args()
    return args


def _main():
    args = _parse_args()

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
    torch.save(hubert.state_dict, str(args.checkpoint_path))


if __name__ == "__main__":
    _main()
