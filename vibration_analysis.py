import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))

def peak(x):
    return float(np.max(np.abs(x)))

def peak_to_peak(x):
    return float(np.max(x) - np.min(x))

def crest_factor(x):
    return peak(x) / max(rms(x), 1e-12)

def kurtosis(x):
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std(ddof=0)
    return float(np.mean((x - m)**4) / (s**4)) if s else 0.0

def skewness(x):
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std(ddof=0)
    return float(np.mean((x - m)**3) / (s**3)) if s else 0.0

def fft_spectrum(x, fs, window='hann'):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)

    if window == 'hann':
        w = np.hanning(n)
    else:
        w = np.ones(n)

    X = np.abs(rfft(x * w)) / (n / 2)
    X[0] /= 2
    f = rfftfreq(n, 1 / fs)

    return f, X

def envelope_spectrum(x, fs, band=(1000, 5000), order=4):
    b, a = butter(
        order,
        [band[0] / (fs / 2), band[1] / (fs / 2)],
        btype='bandpass'
    )

    xf = filtfilt(b, a, x)
    env = np.abs(hilbert(xf))
    env = env - env.mean()

    return fft_spectrum(env, fs)

def bearing_fault_frequencies(rpm, z, d, D, contact_angle_deg=0):
    fs = rpm / 60.0
    a = np.deg2rad(contact_angle_deg)
    c = np.cos(a)

    return {
        'shaft_Hz': fs,
        'BPFO_Hz': z / 2 * fs * (1 - d / D * c),
        'BPFI_Hz': z / 2 * fs * (1 + d / D * c),
        'BSF_Hz': D / (2 * d) * fs * (1 - (d / D * c)**2),
        'FTF_Hz': fs / 2 * (1 - d / D * c)
    }

def gear_fault_frequencies(rpm, teeth, shaft_ratio=1.0):
    fs = rpm / 60.0 * shaft_ratio
    gmf = teeth * fs

    return {
        'shaft_Hz': fs,
        'GMF_Hz': gmf,
        'GMF_harmonics_Hz': [k * gmf for k in range(1, 6)],
        'sidebands_Hz': [
            gmf + k * fs for k in range(-3, 4) if k != 0
        ]
    }

def match_peaks(freqs, amps, targets, tol_hz=2.0, top=10):
    idx = np.argsort(amps)[::-1][:top]
    out = []

    for i in idx:
        for name, t in targets.items():
            if abs(freqs[i] - t) <= tol_hz:
                out.append({
                    'frequency_Hz': float(freqs[i]),
                    'amplitude': float(amps[i]),
                    'fault': name,
                    'target_Hz': float(t)
                })

    return sorted(out, key=lambda r: -r['amplitude'])

def waterfall(spectra, freqs, labels):
    return {
        'frequencies': freqs,
        'spectra': spectra,
        'labels': labels
    }

@dataclass
class VibrationAnalysis:
    x: np.ndarray
    fs: float
    rpm: float

    def time_features(self):
        return {
            'RMS': rms(self.x),
            'Peak': peak(self.x),
            'Peak_to_Peak': peak_to_peak(self.x),
            'Crest_Factor': crest_factor(self.x),
            'Kurtosis': kurtosis(self.x),
            'Skewness': skewness(self.x)
        }

    def spectrum(self):
        return fft_spectrum(self.x, self.fs)

    def envelope(self, band=(1000, 5000)):
        return envelope_spectrum(self.x, self.fs, band)

    def bearing_targets(self, z, d, D, angle=0):
        return bearing_fault_frequencies(self.rpm, z, d, D, angle)

    def gear_targets(self, teeth, ratio=1.0):
        return gear_fault_frequencies(self.rpm, teeth, ratio)

    def diagnose_bearing(self, z, d, D, band=(1000, 5000), tol=2.0):
        f, a = self.envelope(band)
        t = self.bearing_targets(z, d, D)
        return match_peaks(f, a, t, tol)

    def diagnose_gear(self, teeth, ratio=1.0, tol=2.0):
        f, a = self.spectrum()
        t = gear_fault_frequencies(self.rpm, teeth, ratio)

        targets = {'GMF': t['GMF_Hz']}

        for i, h in enumerate(t['GMF_harmonics_Hz'], 1):
            targets[f'GMF_{i}x'] = h

        for s in t['sidebands_Hz']:
            targets[f'Sideband_{s:.2f}Hz'] = s

        return match_peaks(f, a, targets, tol)