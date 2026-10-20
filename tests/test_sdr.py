import numpy as np
import pytest
import soundfile as sf

from fm_project.channel import measured_snr_db
from fm_project.iqio import FileIQSink, FileIQSource, MemoryIQSink, SyntheticIQSource
from fm_project.receiver import BroadcastFMReceiver
from fm_project.sdr import (ComplexFMModulator, FMDiscriminator, IQChannel,
                            IntegerDecimator, IntegerInterpolator)
from fm_project.sdr_cli import main


FS = 240000
DF = 75000


def message(n=12000, fs=FS):
    t = np.arange(n)/fs
    return 0.4*np.sin(2*np.pi*1100*t) + 0.2*np.sin(2*np.pi*2300*t)


@pytest.mark.parametrize("block", [64, 128, 256, 512, 1024])
def test_fm_block_continuity_and_recovery(block):
    m = message()
    whole = ComplexFMModulator(FS, DF).process(m)
    tx = ComplexFMModulator(FS, DF)
    chunks = np.concatenate([tx.process(m[i:i+block]) for i in range(0, len(m), block)])
    assert np.max(np.abs(whole-chunks)) < 1e-10
    for method in ("product", "unwrap"):
        rx = FMDiscriminator(FS, DF, method=method)
        restored = np.concatenate([rx.process(chunks[i:i+block])
                                   for i in range(0, len(chunks), block)])
        assert np.max(np.abs(restored[1:]-m[1:])) < 1e-10
        assert restored[0] == 0


def test_reset_and_invalid_parameters():
    m = message(1000)
    tx = ComplexFMModulator(FS, DF, initial_phase=0.2)
    a = tx.process(m)
    tx.reset()
    assert np.allclose(a, tx.process(m))
    with pytest.raises(ValueError):
        tx.process(np.array([np.nan]))
    with pytest.raises(ValueError):
        tx.process(np.array([1.1]))
    with pytest.raises(ValueError):
        ComplexFMModulator(FS, FS/2)
    with pytest.raises(ValueError):
        FMDiscriminator(FS, DF, method="unknown")


def test_channel_snr_cfo_and_streaming():
    z = ComplexFMModulator(FS, DF).process(message())
    channel = IQChannel(FS, snr_db=15, seed=3)
    noisy = channel.process(z)
    assert abs(measured_snr_db(z, noisy)-15) < 0.2
    clean = IQChannel(FS, cfo_hz=5000, initial_phase=0.4,
                      bandwidth_hz=180000, amplitude=0.7)
    whole = clean.process(z)
    clean.reset()
    chunks = np.concatenate([clean.process(z[i:i+257]) for i in range(0, len(z), 257)])
    assert np.max(np.abs(whole-chunks)) < 1e-10
    assert abs(np.sqrt(np.mean(np.abs(whole[100:])**2))-0.7) < 0.03


def test_resampler_streaming_and_audio_recovery():
    fs = 48000
    t = np.arange(fs//4)/fs
    audio = 0.5*np.sin(2*np.pi*1000*t) + 0.2*np.sin(2*np.pi*2000*t)
    up = IntegerInterpolator(fs, 5)
    whole_up = up.process(audio)
    up.reset()
    chunk_up = np.concatenate([up.process(audio[i:i+133]) for i in range(0, len(audio), 133)])
    assert np.allclose(whole_up, chunk_up, atol=1e-12)
    iq = ComplexFMModulator(FS, DF).process(whole_up)
    recovered = FMDiscriminator(FS, DF).process(iq)
    down = IntegerDecimator(FS, 5, cutoff_hz=15000)
    whole_down = down.process(recovered)
    down.reset()
    chunk_down = np.concatenate([down.process(recovered[i:i+257])
                                 for i in range(0, len(recovered), 257)])
    assert np.allclose(whole_down, chunk_down, atol=1e-12)
    assert len(whole_down) == len(audio)
    # Both 51-tap causal filters contribute five audio-sample delays.
    assert np.sqrt(np.mean((whole_down[200+10:-200]-audio[200:-210])**2)) < 0.002


def test_file_and_mock_iq_adapters(tmp_path):
    z = ComplexFMModulator(FS, DF).process(message(1000)).astype(np.complex64)
    path = tmp_path / "sent.iq"
    with FileIQSink(path, sample_rate=FS, center_freq=99.5e6) as sink:
        sink.write(z[:300])
        sink.write(z[300:])
    with FileIQSource(path) as source:
        assert source.sample_rate == FS and source.fmt == "cf32le"
        restored = np.concatenate([source.read(127) for _ in range(8)])
        assert np.array_equal(restored[:1000], z)
    for fmt, dtype, scale in (("cs8", "i1", 128), ("ci16le", "<i2", 32768)):
        raw = tmp_path / f"{fmt}.iq"
        np.array([10, -20, 30, -40], dtype=dtype).tofile(raw)
        with FileIQSource(raw, sample_rate=FS, fmt=fmt) as source:
            assert np.allclose(source.read(2), [10/scale-20j/scale, 30/scale-40j/scale])
    source = SyntheticIQSource(z, FS)
    sink = MemoryIQSink(FS)
    while True:
        block = source.read(123)
        if not len(block):
            break
        sink.write(block)
    assert np.array_equal(sink.result(), z)


def test_broadcast_receiver_blocks_match():
    z = ComplexFMModulator(FS, DF).process(message(12000))
    rx = BroadcastFMReceiver(FS, decimation=5, channel_bandwidth_hz=180000)
    whole = rx.process(z)
    rx.reset()
    chunked = np.concatenate([rx.process(z[i:i+257]) for i in range(0, len(z), 257)])
    assert np.allclose(whole, chunked, atol=1e-10)
    assert len(whole) == len(z)//5
    assert np.all(np.isfinite(whole))


def test_cli_wav_and_iq_roundtrip(tmp_path):
    t = np.arange(4800)/48000
    audio = 0.5*np.sin(2*np.pi*1000*t)
    source = tmp_path / "input.wav"
    output = tmp_path / "output.wav"
    iq = tmp_path / "signal.iq"
    sf.write(source, audio, 48000)
    assert main(["simulate", "--input", str(source), "--output", str(output),
                 "--iq-out", str(iq), "--block-size", "128"]) == 0
    y, rate = sf.read(output)
    assert rate == 48000 and len(y) == len(audio)
    with FileIQSource(iq) as saved:
        assert saved.sample_rate == FS and len(saved.read(25000)) == 24000
    original = source.read_bytes()
    assert main(["simulate", "--input", str(source), "--output", str(output),
                 "--iq-out", str(source)]) == 1
    assert source.read_bytes() == original
    saved_iq = iq.read_bytes()
    assert main(["receive", "--input", str(iq), "--output", str(iq)]) == 1
    assert iq.read_bytes() == saved_iq
