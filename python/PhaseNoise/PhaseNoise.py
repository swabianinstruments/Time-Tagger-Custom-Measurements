import TimeTagger as TT
import numpy as np
from scipy.signal import get_window
import numba
import rocket_fft  # Required for numba.njit support for numpy.fft

samples_per_octave = 64
NFFT = samples_per_octave * 4
octaves = 32  # 10 MHz <-> 1 mHz


class PhaseNoise(TT.CustomMeasurement):
    """
    This measurement implements a quasi-logarithmic phase noise estimator.
    Please use Tagger.setSoftwareClock for the reference clock if the built-in reference is not good enough.

    As power spectral density estimator, the Welch's method is used:
    * all time samples are split up in groups of NFFT samples
    * a linear regression error is removed from the time sample groups as demodulator for phase samples
    * each group is multiplied with a Hann window for smoother results between blocks
    * the squared absolute of the FFT of each group is averaged over time
    * to overcome the window preference on the inner samples, the groups are overlapping by half the samples

    This method provides a linear PSD estimation. For the quasi-logarithmic frequency behavior, only the upper half of the spetrum is used.
    For the lower half of the spectrum, each group of two time samples is averaged and recursively used for this whole method.
    This method yields NFFT/4 linear distributed spectral samples per octave.
    """

    # Numeric implementation
    # Accelerated with numba
    @numba.experimental.jitclass
    class Kernel:
        channel: numba.int32
        window: numba.float64[::1]
        first_sample: numba.int64
        first_period: numba.int64
        sample_counter: numba.int64
        traces: numba.float64[:, ::1]
        traces_utilization: numba.int64[::1]
        PSDs: numba.float64[:, ::1]
        PSDs_count: numba.int64[::1]

        def __init__(self, channel, window):
            self.channel = channel
            self.window = window
            self.clear()

        def clear_traces(self):
            self.first_sample = 0
            self.first_period = 0
            self.sample_counter = 0
            self.traces = np.zeros((octaves, NFFT), dtype=np.float64)
            self.traces_utilization = np.zeros((octaves,), dtype=np.int64)

        def clear(self):
            self.clear_traces()
            self.PSDs = np.zeros((octaves, NFFT//2 + 1), dtype=np.float64)
            self.PSDs_count = np.zeros((octaves,), dtype=np.int64)

        def push(self, tags):
            for tag in tags:
                if tag['type'] != TT.TagType.TimeTag:
                    # On overflows, just clear the traces and keep the PSDs
                    if tag['type'] == TT.TagType.OverflowEnd or tag['type'] == TT.TagType.Error:
                        self.clear_traces()
                elif tag['channel'] == self.channel:
                    # Fast linear correction: Demodulate with the first measured period
                    # Required as unsigned integer with defined overflow behavior
                    if self.sample_counter == 0:
                        self.first_sample = tag['time']
                    elif self.sample_counter == 1:
                        self.first_period = tag['time'] - self.first_sample
                    rel_time = tag['time'] - \
                        self.first_period * self.sample_counter
                    self.sample_counter += 1

                    # Store this phase sample in the buffer
                    self.traces[0, self.traces_utilization[0]] = rel_time
                    self.traces_utilization[0] += 1

                    # For all full phase traces
                    for i in range(octaves):
                        if self.traces_utilization[i] != NFFT:
                            break
                        t = self.traces[i]

                        if True:
                            # Linear regression
                            x = np.arange(NFFT)
                            t = t - x * (12 * np.sum(x*t) - 6 * (NFFT - 1)
                                         * np.sum(t)) / (NFFT * (NFFT * NFFT - 1)) - t[0]

                        # Sum up the PSD of this phase trace
                        self.PSDs[i] += np.abs(np.fft.rfft(t * self.window))**2
                        self.PSDs_count[i] += 1

                        if i+1 < octaves and self.traces_utilization[i+1] == 0:
                            # The lower half has not been averaged down, do it now
                            # This is only done once for each octave
                            for k in range(NFFT // 4):
                                self.traces[i+1, self.traces_utilization[i+1] +
                                            k] = self.traces[i, 2*k] + self.traces[i, 2*k+1]
                            self.traces_utilization[i+1] += NFFT // 4

                        # 50% overlapping approach: Move upper half of the trace to lower half
                        self.traces_utilization[i] = NFFT // 2
                        self.traces[i, :NFFT//2] = self.traces[i, NFFT//2:]

                        # Stop on the last octave
                        # TODO Use better memory management and just reallocate the buffers
                        if i+1 == octaves:
                            break

                        # Average even+odd sample to one sample for the next octave
                        for k in range(NFFT // 4):
                            self.traces[i+1, self.traces_utilization[i+1] +
                                        k] = self.traces[i, 2*k] + self.traces[i, 2*k+1]
                        self.traces_utilization[i+1] += NFFT // 4

    def __init__(self, tagger, channel, freq):
        TT.CustomMeasurement.__init__(self, tagger)
        self.register_channel(channel=channel)
        self.freq = freq

        window = get_window('hann', NFFT)
        window *= 1 / np.mean(window)
        self.kernel = self.Kernel(channel, window)

        self.finalize_init()

    def __del__(self):
        self.stop()

    def process(self, incoming_tags, begin_time, end_time):
        self.kernel.push(incoming_tags)

    def getData(self):
        with self.mutex:
            f = np.array([])
            dBcHz = np.array([])
            for i in range(octaves)[::-1]:
                # Skip octaves without data
                if self.kernel.PSDs_count[i] == 0:
                    continue

                # Skip the lower half of all but the first octave
                lower_index = 1 if len(f) == 0 else NFFT // 4

                scale_PSD = (1e-12 * 2*np.pi*self.freq)**2 / \
                    (self.kernel.PSDs_count[i] * NFFT * self.freq * 2**i)
                f = np.concatenate(
                    (f, (self.freq / NFFT / 2**i) * np.arange(NFFT // 2 + 1)[lower_index: -1]))
                dBcHz = np.concatenate(
                    (dBcHz, 10 * np.log10(scale_PSD * self.kernel.PSDs[i, lower_index: -1])))

            return f, dBcHz


if __name__ == '__main__':
    import time
    if True:
        # Real Time Tagger hardware
        f0 = 10e6
        ch1 = 1
        ch2 = 5
        t = TT.createTimeTagger('', 2)
        t.setHardwareBufferSize(500e6)
        if False:
            # TODO Unreleased feature: On-device average of the rising and falling edge
            t.setInputDelay(-ch1, round(0.5 * 1e12 / f0))
            t.setInputDelay(-ch2, round(0.5 * 1e12 / f0))
            t.xtra_setAvgRisingFalling(ch1, True)
            t.xtra_setAvgRisingFalling(ch2, True)
        t.setSoftwareClock(ch2)
        pn = PhaseNoise(t, ch1, f0)
    else:
        # Simulated data from the Time Tagger Virtual: 9ps RMS white phase noise + 1ps RMS period error
        t = TT.createTimeTaggerVirtual()
        t.setReplaySpeed(1)
        t.setTestSignal(1, True)
        pn = PhaseNoise(t, 1, 8e5)
    start_time = time.clock_gettime(time.CLOCK_MONOTONIC)

    # Plot the phase noise diagram
    import matplotlib.pyplot as plt
    while True:
        plt.pause(1)
        plt.clf()
        plt.semilogx(*pn.getData())
        now_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        plt.title(
            f'Capture duration {1e-12*pn.getCaptureDuration():.2f}, real time {now_time - start_time:.2f}')
        plt.xlabel('Frequency offset (Hz)')
        plt.ylabel('PSD of phase noise (dBc/Hz)')
        plt.grid(True)
        plt.grid(which='minor', axis='x', linestyle=':')
