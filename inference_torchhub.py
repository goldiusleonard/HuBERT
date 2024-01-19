import torch, torchaudio

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()

# Load audio
wav, sr = torchaudio.load(
    "F:/Study/Language Model/Voice Changer/audio-dataset/audio_sample/male.wav"
)
# assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(wav)
