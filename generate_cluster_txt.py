import json

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="F:/Study/Language Model/Voice Changer/audio-dataset/LibriSpeech/wavs/train-clean-100",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default="F:/Study/Language Model/Voice Changer/audio-dataset/LibriSpeech/cluster-train-list.txt",
    )
    args = parser.parse_args()
    return args


def _get_audio_paths(
    data_root: str, audio_exts: list = [".wav", ".flac", ".mp3", ".aac"]
):
    audio_paths = []

    for audio_ext in audio_exts:
        audio_paths.extend(list(Path(data_root).rglob(f"*{audio_ext}")))

    return list(set(audio_paths))


def _main():
    args = _parse_args()

    audio_paths = _get_audio_paths(args.data_root)

    Path(args.txt_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.txt_path, "w") as f:
        for audio_path in tqdm(audio_paths):
            audio_path_cut = Path(
                *Path(audio_path).parts[len(Path(args.data_root).parts) - 1 :]
            )
            audio_path_cut = audio_path_cut.parent / audio_path_cut.stem
            f.write(str(audio_path_cut).replace("\\", "/"))
            f.write("\n")


if __name__ == "__main__":
    _main()
