# External broadcast IQ recording

The local file `fm_rds_250k_1Msamples.iq` comes from a [public repository referenced by the PySDR tutorial](https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq). The [tutorial](https://www.pysdr.org/content/rds) describes a recording of a 99.5 MHz station with a 250 kHz IQ sample rate and 1,000,000 `complex64` samples. Its 8,000,000-byte length represents four seconds. The source repository does not state an explicit redistribution license for the recording, so the raw IQ is kept locally and excluded from Git.

Download it from the project root in Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/real | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/777arc/498x/master/fm_rds_250k_1Msamples.iq' -OutFile data/real/fm_rds_250k_1Msamples.iq
```

SHA-256 of the file used here: `c336b7be8e08081521465916921bf995253701eb84d218cf73a02dde958a890b`. This hash identifies the external validation sample. If the upstream file changes, recheck its format and the resulting measurements. Run the receiver with:

```powershell
python -m fm_project receive --input data/real/fm_rds_250k_1Msamples.iq --iq-rate 250000 --format cf32le --center-freq 99500000 --decimation 5 --audio-gain 0.5 --output results/sdr/real_fm_recovered.wav --report results/sdr/real_fm_report.json --figure results/figures/real_fm_receiver.png
```

This is a third-party recording of a real RF signal. Files under `data/generated/` are outputs of this project's software simulation. The command above runs `fm_project.receiver.BroadcastFMReceiver`; it does not copy the tutorial's recovered audio.
