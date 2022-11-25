import TimeTagger
import numpy as np
from scipy.stats import linregress
from scipy import signal
import numba                    # Required for fast processing of many events
import allantools               # ADEV + MDEV + HDEV
import matplotlib.pyplot as plt  # Plotting library

# This is the custom measurement
# The script for its usage is below, so please scroll down


class PhaseInterpolator(TimeTagger.CustomMeasurement):
    """
    Custom measurement written in Python.
    Please see the CustomStartStop example for more documentation about the API of custom measurements.
    :param tagger: The instance of the TimeTagger
    :type tagger: TimeTaggerBase
    :param channels: List of channels for which the phase shall be recorded
    :type channels: int[]
    :param rate: Sampling rate of the output phase data in Hz
    :type rate: float
    :param norm_freq: Norminal input frequency of each channel in Hz
    :type norm_freq: float[]
    :param block_size: Internal buffer size for the phase data
    :type block_size: int
    """

    def __init__(self, tagger, channels, rate, norm_freq, block_size=16*1024*1024):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.channels = np.array(channels, dtype=np.int32)
        self.period = np.int64(1e12 / rate)
        self.block_size = block_size
        self.norm_freq = np.array(norm_freq)
        self.average_tags = np.maximum(np.round(
            self.norm_freq / (2.0 * rate)), np.ones(len(channels))).astype(np.int32)
        self.phase_offset = np.array(
            norm_freq, dtype=np.float64) / (rate * self.average_tags)

        for chan in channels:
            self.register_channel(channel=chan)

        self.clear_impl()
        self.finalize_init()

    def __del__(self):
        self.stop()

    def getData(self, clear=False):
        """
        Fetches the phase data
        :param clear: Removes the fetched data from the internal buffer. If set, every phase sample is returned exactly once.
        :type clear: bool
        :returns: The phase offset in seconds
        :rtype: float[]
        """
        with self.mutex:
            i_min = np.min(self.next_index)
            i_max = np.max(self.next_index)

            data = self.phase_buffer[:i_min] * \
                (self.average_tags / self.norm_freq)

            if clear:
                # Move remaining partial data
                self.phase_buffer[:i_max -
                                  i_min] = self.phase_buffer[i_min:i_max]
                self.next_index -= i_min

            return data

    def clear_impl(self):
        self.phase_buffer = np.zeros(
            [self.block_size, len(self.channels)], dtype=np.float64)
        self.current_phase = np.zeros(len(self.channels), dtype=np.float64)
        self.last_timestamps = np.zeros(len(self.channels), dtype=np.int64)
        self.next_timestamps = np.zeros(len(self.channels), dtype=np.int64)
        self.next_index = np.zeros(len(self.channels), dtype=np.uintp)
        self.on_start()

    def on_start(self):
        self.initialized = np.zeros(len(self.channels), dtype=bool)
        self.next_phase_sample = np.zeros(len(self.channels), dtype=np.int64)
        self.next_timestamps_count = np.zeros(
            len(self.channels), dtype=np.int32)

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def fast_process(tags, initialized, channels, next_phase_sample, next_index, block_size, phase_buffer, current_phase, last_timestamps, period, phase_offset, next_timestamps, next_timestamps_count, average_tags):
        initialized_all = np.all(initialized)
        for tag in tags:
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # TODO overflow handling, return NaN for unknown samples
                assert False, "overflow behavior is not implemented, please use setEventDivider to reduce the data rate"
            else:
                for i, chan in enumerate(channels):
                    if chan == tag['channel']:

                        # Rectangular low pass and down sampling filter on the input timestamps
                        next_timestamps[i] += tag['time']
                        next_timestamps_count[i] += 1
                        if next_timestamps_count[i] != average_tags[i]:
                            continue
                        tag_time = next_timestamps[i]
                        next_timestamps_count[i] = 0
                        next_timestamps[i] = 0

                        if initialized_all:

                            while np.int64(next_phase_sample[i] - tag_time) < 0:
                                # TODO implement a "full"-state which discards samples in a defined way
                                assert next_index[i] < block_size, "buffer overflow, please increase block_size or poll more frequently"

                                # linear interpolation between the last and this tag
                                phase_buffer[next_index[i], i] = current_phase[i] + np.float64(
                                    next_phase_sample[i] - last_timestamps[i]) / np.float64(tag_time - last_timestamps[i])

                                # state advancing per phase sample
                                next_index[i] += 1
                                next_phase_sample[i] += period * \
                                    average_tags[i]
                                current_phase[i] -= phase_offset[i]

                        # state advancing per tag
                        current_phase[i] += 1
                        last_timestamps[i] = tag_time

                        # initial condition: Wait until there was an event on all input channels
                        if not initialized[i]:
                            initialized[i] = True
                            current_phase[i] = 0.0
                            initialized_all = np.all(initialized)
                            if initialized_all:
                                next_phase_sample += tag_time

    def process(self, incoming_tags, begin_time, end_time):
        # Call the numba compiled method, for a HUGE performance gain
        self.fast_process(incoming_tags, self.initialized, self.channels, self.next_phase_sample, self.next_index,
                          self.block_size, self.phase_buffer, self.current_phase, self.last_timestamps, self.period, self.phase_offset, self.next_timestamps, self.next_timestamps_count, self.average_tags)


#################################
### PLEASE START READING HERE ###
#################################
if __name__ == '__main__':

    # The rate at which the phase of all inputs are sampled
    sampling_rate = 1e2

    # Channels on which the phase is measured
    channels = [1, 2]

    # Norminal input frequency, which is removed of the phase slope, after the event divider
    norminal_freq = [10e6, 10e6]

    # Trigger level for every input, default is 0.5V
    trigger_levels = [0.5, 0.5]

    # On-device event divider, usful to limit the USB and CPU resources, use "1" if no divider shall be used
    event_divider = [10, 10]

    # Create TimeTagger and apply the settings
    with TimeTagger.createTimeTagger() as tagger:

        # Software clock on external reference
        # tagger.setEventDivider(channel=8, divider=10)
        # tagger.setSoftwareClock(input_channel=8, input_frequency=1e6) # Note: freq after the divider

        for c, v in zip(channels, trigger_levels):
            tagger.setTriggerLevel(channel=c, voltage=v)

        for c, d in zip(channels, event_divider):
            tagger.setEventDivider(channel=c, divider=d)

        # Create the custom measurement
        pi = PhaseInterpolator(
            tagger=tagger,
            channels=channels,
            rate=sampling_rate,
            norm_freq=np.array(norminal_freq) / event_divider
        )

        # Main loop
        while True:
            # Fetches the data
            # IMPORTANT: If you want this measurement to run for a long time,
            #            you need to call it with clear=True and to keep the valid
            #            phase samples locally.
            phase_all = pi.getData(clear=False)

            # Save everything to disk
            np.savetxt('phase_data.txt', phase_all)

            if phase_all.shape[0] <= 4:
                # no data? So nothing to plot
                plt.pause(1)
                continue

            # Plot everything
            plt.clf()
            for i, phase in enumerate(phase_all.T):
                # Plot allan-alike graphs
                plt.subplot(3, phase_all.shape[1], i+1)
                res = allantools.adev(phase, rate=sampling_rate)
                plt.loglog(res[0], res[1], label="ADEV")
                res = allantools.mdev(phase, rate=sampling_rate)
                plt.loglog(res[0], res[1], label="MDEV")
                res = allantools.hdev(phase, rate=sampling_rate)
                plt.loglog(res[0], res[1], label="HDEV")
                plt.xlabel('tau (s)')
                plt.ylabel('sigma (1)')
                plt.legend()
                plt.grid(True)

                # Phase error
                plt.subplot(3, phase_all.shape[1], i+1+1*phase_all.shape[1])
                x = np.arange(phase.size) / sampling_rate
                res = linregress(x, phase)
                plt.plot(x, phase - (res.slope*x + res.intercept),
                         label=f"phase + {-res.slope:.4g} * t + {-res.intercept:.4g} s")
                plt.xlabel('time (s)')
                plt.ylabel('normalized phase drift (s)')
                plt.legend()
                plt.grid(True)

                # Frequency error
                plt.subplot(3, phase_all.shape[1], i+1+2*phase_all.shape[1])
                for label, lp in [
                    ('original', np.array([1])),
                    ('.1 cutoff', signal.firwin(25, 0.1)),
                    ('.01 cutoff', signal.firwin(251, 0.01)),
                ]:
                    if len(phase) <= len(lp):
                        continue
                    # LP on phase, not frequency!
                    dphase = np.diff(np.convolve(phase, lp, 'valid'))
                    df = (-norminal_freq[i] * dphase) / \
                        (dphase + 1.0 / sampling_rate)
                    x = (np.arange(df.size) + 0.5 *
                         (phase.size - df.size)) / sampling_rate
                    plt.plot(
                        x, df, label=f"frequency {label} - {norminal_freq[i]:.4g} Hz")
                plt.xlabel('time (s)')
                plt.ylabel('frequency offset (Hz)')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.show(block=False)
            plt.draw()
            plt.pause(1)
