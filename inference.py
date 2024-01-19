import torch
import torchaudio

from argparse import ArgumentParser
from hubert.model import HubertSoft, HubertDiscrete
from collections import OrderedDict


def _parse_args():
    parser = ArgumentParser(description="")

    parser.add_argument(
        "--model_type",
        type=str,
        help="choose hubert model type (`soft` or `discrete`)",
        default="soft",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="specify hubert checkpoint to load",
        default="F:/Study/Language Model/Voice Changer/hubert/ckpt/model-best.pt",
    )
    parser.add_argument(
        "--kmeans_ckpt",
        type=str,
        help="specify kmeans checkpoint to load (`none` for HuBERT soft)",
        default="none",
    )
    parser.add_argument(
        "--n_clusters",
        help="number of clusters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--sound_input_path",
        help="specify sound input file path",
        type=str,
        default="F:/Study/Language Model/Voice Changer/audio-dataset/audio_sample/male.wav",
    )

    args = parser.parse_args()

    return args


def _build_model(
    model_type: str,
    checkpoint_path: str,
    kmeans_ckpt: str,
    n_clusters: int,
):
    assert model_type.lower() == "soft" or model_type.lower() == "discrete"

    if model_type.lower() == "soft":
        model = HubertSoft()
    elif model_type.lower() == "discrete":
        assert kmeans_ckpt.lower() != "none"
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans_ckpt_dict = torch.load(kmeans_ckpt)
        kmeans.set_params(kmeans_ckpt_dict)
        model = HubertDiscrete(kmeans)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict()

    for k, v in ckpt["hubert"].items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model


def _main():
    args = _parse_args()
    model = _build_model(
        args.model_type, args.model_ckpt, args.kmeans_ckpt, args.n_clusters
    )

    wav, sr = torchaudio.load(args.sound_input_path)
    wav = wav.unsqueeze(0).cuda()

    units = model.units(wav)
    print(units)


if __name__ == "__main__":
    _main()
