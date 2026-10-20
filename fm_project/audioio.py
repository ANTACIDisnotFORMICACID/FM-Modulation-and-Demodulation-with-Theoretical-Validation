"""可选声卡收发接口；离线 DSP 不依赖声卡设备。

录音与放音使用独立流，避免要求设备支持全双工。缺失 sounddevice 时，
本模块仍可导入；只有访问声卡的函数会抛错。
"""
from __future__ import annotations

import time

import numpy as np

try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:                                     # pragma: no cover
    sd = None
    HAVE_SD = False


# 设备编号会在插拔一分二线后变化，因此 None 表示“运行时使用 Windows 默认端点”。
DEFAULT_OUT_DEV = None
DEFAULT_IN_DEV = None
DEFAULT_FS = 48000
LOOPBACK_MAX_OUTPUT_PEAK = 0.05   # -26 dBFS：直连麦克风口的保守起始上限


def require_sd():
    if not HAVE_SD:
        raise RuntimeError(
            "未安装 sounddevice。声卡功能需要单独执行 pip install sounddevice。"
        )


def list_devices():
    require_sd()
    rows = []
    for i, d in enumerate(sd.query_devices()):
        rows.append(dict(index=i, name=d["name"],
                         max_in=d["max_input_channels"],
                         max_out=d["max_output_channels"],
                         default_sr=int(d["default_samplerate"]),
                         api=sd.query_hostapis(d["hostapi"])["name"]))
    return rows


def resolve_devices(out_device=None, in_device=None, sample_rate=DEFAULT_FS):
    """解析并验证输出/输入端点。

    未显式指定时使用 PortAudio 当前默认设备，避免插拔后旧编号指向
    麦克风阵列、扬声器或方向错误的端点。
    """
    require_sd()
    default_in, default_out = sd.default.device
    out_device = int(default_out if out_device is None else out_device)
    in_device = int(default_in if in_device is None else in_device)
    if out_device < 0 or in_device < 0:
        raise RuntimeError("没有可用的 Windows 默认输入/输出音频端点")

    out_info = sd.query_devices(out_device)
    in_info = sd.query_devices(in_device)
    if int(out_info["max_output_channels"]) < 1:
        raise ValueError(f"设备 {out_device} {out_info['name']!r} 不是输出设备")
    if int(in_info["max_input_channels"]) < 1:
        raise ValueError(f"设备 {in_device} {in_info['name']!r} 不是输入设备")
    sd.check_output_settings(device=out_device, channels=1, samplerate=sample_rate,
                             dtype="float32")
    sd.check_input_settings(device=in_device, channels=1, samplerate=sample_rate,
                            dtype="float32")
    return out_device, in_device


def _checked_peak(x, gain, max_output_peak):
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain 必须是非负有限数")
    peak = float(np.max(np.abs(x))) if np.size(x) else 0.0
    peak *= float(gain)
    if peak > float(max_output_peak) + 1e-7:
        raise ValueError(
            f"输出峰值 {peak:.3f} 超过回环安全起始上限 "
            f"{max_output_peak:.3f}；请降低 gain"
        )
    return peak


def _to_channels(x, nch):
    """转为输出通道数；单声道回环只驱动左声道。

    不把单声道复制到右声道，可减少普通 TRS 公对公线把右声道
    连到麦克风偏置触点时的主动驱动。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == nch:
        return x
    if x.shape[1] > nch:
        return x[:, :nch]
    out = np.zeros((x.shape[0], nch), dtype=np.float32)
    out[:, 0] = x[:, 0]
    return out


def play(x, sample_rate=DEFAULT_FS, *, device=None, gain=1.0,
         lead_silence_s=0.2, tail_silence_s=0.5,
         max_output_peak=LOOPBACK_MAX_OUTPUT_PEAK):
    """播放一段信号（阻塞）。返回实际送入声卡的完整波形（含首尾静默）。"""
    require_sd()
    device, _ = resolve_devices(out_device=device, sample_rate=sample_rate)
    nch = int(sd.query_devices(device)["max_output_channels"])
    nch = max(1, min(2, nch))                     # 兼容单声道与双声道输出
    lead = np.zeros(int(lead_silence_s * sample_rate), dtype=np.float32)
    tail = np.zeros(int(tail_silence_s * sample_rate), dtype=np.float32)
    full = np.concatenate([lead, np.asarray(x, dtype=np.float32).ravel(), tail])
    full = full * float(gain)
    _checked_peak(full, 1.0, max_output_peak)
    sd.play(_to_channels(full, nch), samplerate=sample_rate, device=device, blocking=True)
    return full


def record(duration_s, sample_rate=DEFAULT_FS, *, device=None, channels=1):
    """录制一段（阻塞），返回 (n_samples, channels) 数组和时间轴。"""
    require_sd()
    _, device = resolve_devices(in_device=device, sample_rate=sample_rate)
    maxc = int(sd.query_devices(device)["max_input_channels"])
    channels = max(1, min(int(channels), maxc))
    n = int(duration_s * sample_rate)
    rec = sd.rec(n, samplerate=sample_rate, channels=channels,
                 dtype="float32", device=device, blocking=True)
    return np.asarray(rec, dtype=float), np.arange(n) / sample_rate


def play_and_record(x, sample_rate=DEFAULT_FS, *, out_device=None, in_device=None,
                    in_channels=1, gain=LOOPBACK_MAX_OUTPUT_PEAK, extra_tail_s=0.5,
                    max_output_peak=LOOPBACK_MAX_OUTPUT_PEAK):
    """播放并同时录音。

    某些音频端点组合不支持全双工 playrec，因此这里用两个独立的流实现：
    先启动录音流，再启动播放流，最后一起停止。这样不依赖 PortAudio 的全双工绑定。

    Returns
    -------
    (tx_full, rx) : tx_full 是实际送入声卡的完整波形（含首尾静默），
                    rx 是录到的信号（可能比 tx_full 长）
    """
    require_sd()
    out_device, in_device = resolve_devices(out_device, in_device, sample_rate)

    nch_out = max(1, min(2, int(sd.query_devices(out_device)["max_output_channels"])))
    nch_in = max(1, min(int(in_channels), int(sd.query_devices(in_device)["max_input_channels"])))

    lead = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    tail = np.zeros(int((0.5 + extra_tail_s) * sample_rate), dtype=np.float32)
    raw = np.concatenate([lead, np.asarray(x, dtype=np.float32).ravel(), tail])
    _checked_peak(raw, gain, max_output_peak)
    full = raw * float(gain)

    n_total = len(full)
    rx = np.zeros((n_total + int(0.3 * sample_rate), nch_in), dtype=np.float32)
    pos = {"i": 0}

    def cb(indata, frames, time_info, status):          # noqa: ARG001
        i = pos["i"]
        j = min(i + frames, rx.shape[0])
        if j > i:
            rx[i:j] = indata[: j - i]
        pos["i"] = j

    in_stream = sd.InputStream(device=in_device, samplerate=sample_rate,
                               channels=nch_in, dtype="float32", callback=cb,
                               blocksize=0)
    with in_stream:
        time.sleep(0.15)                                # 让录音先稳定
        sd.play(_to_channels(full, nch_out), samplerate=sample_rate,
                device=out_device, blocking=True)
        time.sleep(0.25)
    return full, rx[: pos["i"]]


def align_by_xcorr(tx, rx, sample_rate, *, up_to_s=1.0):
    """用互相关估计 rx 相对 tx 的延迟（样本数）。

    返回 (delay_samples, peak_corr)。用 tx 的前 up_to_s 作为模板在 rx 中搜索。
    """
    tx = np.asarray(tx, dtype=float).ravel()
    rx = np.asarray(rx, dtype=float)
    if rx.ndim > 1:
        rx = rx[:, 0]
    rx = rx.ravel()
    n = min(len(tx), int(up_to_s * sample_rate))
    tmpl = tx[:n] - np.mean(tx[:n])
    search = rx[: n + int(0.5 * sample_rate)]
    if search.size <= tmpl.size:
        return 0, 0.0
    corr = np.correlate(search - np.mean(search), tmpl, mode="valid")
    k = int(np.argmax(np.abs(corr)))
    denom = np.linalg.norm(tmpl) * max(1e-12, np.linalg.norm(search[k:k + tmpl.size]))
    return k, float(np.max(np.abs(corr)) / denom)


def save_wav(path, x, sample_rate):
    """保存为 16-bit PCM WAV（不依赖 scipy.io，避免版本差异）。"""
    import wave
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / max(1.0, peak)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(x.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return path


def load_wav(path):
    """读取 16-bit PCM WAV，返回 (float 数组, sample_rate)。"""
    import wave
    with wave.open(path, "rb") as fh:
        nch, sw, sr, nfr = (fh.getnchannels(), fh.getsampwidth(),
                            fh.getframerate(), fh.getnframes())
        raw = fh.readframes(nfr)
    if sw != 2:
        raise ValueError("仅支持 16-bit PCM WAV，当前 sampwidth=%d" % sw)
    data = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch)
    return data, sr
