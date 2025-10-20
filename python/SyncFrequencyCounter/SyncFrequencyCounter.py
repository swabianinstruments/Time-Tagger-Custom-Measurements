import TimeTagger
import numpy as np
import numba
from numba import types
from numba.typed import List

class SyncFrequencyCounter(TimeTagger.CustomMeasurement):

    @numba.experimental.jitclass
    class Kernel:

        sync_channel: numba.uint8                   # channel index used as sync trigger
        channels: numba.uint8[::1]                  # array of measurement channel indices
        n_channels: numba.int64                     # number of signals
        fitting_window: numba.uint64                # fitting window in ps

        _sum_delta_time: numba.int64[::1]           # sum of relative tag times for regression
        _sum_i_delta_time: numba.int64[::1]         # sum of tag times * index for regression
        counts_per_channel: numba.int64[::1]        # number of tags in current window per channel

        _cumulative_counts: numba.int64[::1]        # cumulative cycle counts per channel
        _current_counts: numba.int64[::1]           # cycle counts from current window (for avg freq)
        _last_counts: numba.int64[::1]              # cycle counts from previous window (for avg freq)
        _last_fractional_phase: numba.float64[::1]  # fractional phase from previous window (for avg freq)
        _is_initialized: numba.boolean[::1]         # flags: has each channel seen a valid window yet?

        _current_sync: numba.uint64                 # timestamp of current sync event
        _last_sync: numba.uint64                    # timestamp of last sync event

        _win_first_edge_time_ps: numba.uint64[::1]  # first event within the fitting window
        _win_last_edge_time_ps: numba.uint64[::1]   # last event within the fitting window

        _out_sync_times: types.ListType(numba.uint64)                 # output: list of sync trigger times
        _out_counts: types.ListType(numba.uint64)                     # output: integer cycle counts
        _out_fractional_phase: types.ListType(numba.float64)          # output: fractional part of phase
        _out_phase: types.ListType(numba.float64)                     # output: total phase
        _out_frequency: types.ListType(numba.float64)                 # output: average frequency
        _out_instantaneous_frequency: types.ListType(numba.float64)   # output: instantaneous frequency
        _out_is_overflow: types.ListType(numba.uint8)                 # output: overflow mask

        _bad_window: numba.boolean    # flag to control the fitting windows is smaller than the sampling interval


        def __init__(self, channels, sync_channel, fitting_window):

            self.channels = channels
            self.n_channels = channels.shape[0]
            self.sync_channel = sync_channel
            self.fitting_window = fitting_window

            # allocate per-channel accumulators
            n = self.n_channels
            self._sum_delta_time = np.zeros(n, dtype=np.int64)
            self._sum_i_delta_time = np.zeros(n, dtype=np.int64)
            self.counts_per_channel = np.zeros(n, dtype=np.int64)

            # state variables for each channel
            self._cumulative_counts = np.zeros(n, dtype=np.int64)
            self._current_counts = np.zeros(n, dtype=np.int64)
            self._last_counts = np.zeros(n, dtype=np.int64)
            self._last_fractional_phase = np.zeros(n, dtype=np.float64)
            self._is_initialized = np.zeros(n, dtype=np.bool_)
            self._current_sync = np.uint64(0)
            self._last_sync = np.uint64(0)

            # events needed to estimate the frequency for avoiding integer overflow
            self._win_first_edge_time_ps = np.zeros(n, dtype=np.uint64)
            self._win_last_edge_time_ps  = np.zeros(n, dtype=np.uint64)

           # prepare output buffers as typed lists
            self._out_sync_times = List.empty_list(numba.uint64)
            self._out_counts = List.empty_list(numba.uint64)
            self._out_fractional_phase = List.empty_list(numba.float64)
            self._out_phase = List.empty_list(numba.float64)
            self._out_frequency = List.empty_list(numba.float64)
            self._out_instantaneous_frequency = List.empty_list(numba.float64)
            self._out_is_overflow = List.empty_list(numba.uint8)

            self._bad_window = False

        def push(self, tags):

            n_channels = self.n_channels
            is_overflow = False

            # Iterate over each incoming tag
            for tag in tags:
                # Handle tags different from TimeTag tags
                if tag['type'] != TimeTagger.TagType.TimeTag:
                    if tag['type'] == TimeTagger.TagType.Error:
                        raise TypeError("Error event detected.")
                    elif tag['type'] == TimeTagger.TagType.OverflowBegin:
                        is_overflow = True
                    elif tag['type'] == TimeTagger.TagType.MissedEvents:
                        missed_counts = tag["missed_events"]
                        ch = tag["channel"]
                        for j in range(n_channels):
                            if self.channels[j] == ch:
                                self._cumulative_counts[j] += missed_counts
                    continue

                # If this tag is on the sync channel, flush the current window
                if tag['channel'] == self.sync_channel:
                    self._out_sync_times.append(np.uint64(tag["time"]))
                    # First ever sync, update the last sync
                    if self._current_sync == 0:
                        self._current_sync = tag["time"]
                        is_overflow = False
                    else:
                        for j in range(n_channels):
                            # number of tags in this window on channel j
                            counts = self.counts_per_channel[j]
                            # If overflow or too few tags, no regression
                            if is_overflow or counts <= 1:
                                fractional_phase = 0.0
                                instantaneous_frequency = 0.0
                            else:
                                # For integer stability, provide an estimate and substract it
                                N_over_f_estimate = self._win_last_edge_time_ps[j] - self._win_first_edge_time_ps[j]

                                denominator = counts * counts - 1
                                numerator = 6 * ((counts - 1) * self._sum_delta_time[j] - 2 * self._sum_i_delta_time[j]) \
                                            - N_over_f_estimate * denominator
                                N_over_f_offset = numerator / denominator
                                N_over_f = N_over_f_estimate + N_over_f_offset
                                instantaneous_frequency = 1e12 * counts / N_over_f

                                timeoffset_estimate = 2 * self._sum_delta_time[j] - N_over_f_estimate * (counts - 1)
                                fractional_phase = 1.0 - (
                                    (timeoffset_estimate - N_over_f_offset * (counts - 1)) *
                                    0.5 * instantaneous_frequency * 1e-12 / counts
                                )
                            # emit outputs
                            self._out_counts.append(np.uint64(self._current_counts[j]))
                            if is_overflow or counts <= 1:
                                self._out_instantaneous_frequency.append(np.nan)
                                self._out_fractional_phase.append(np.nan)
                                self._out_phase.append(np.nan)
                                self._out_is_overflow.append(np.uint8(is_overflow))
                            else:
                                self._out_fractional_phase.append(np.float64(fractional_phase))
                                self._out_phase.append(np.float64(self._current_counts[j] + fractional_phase))
                                self._out_is_overflow.append(np.uint8(is_overflow))
                                self._out_instantaneous_frequency.append(np.float64(instantaneous_frequency))

                            # average frequency computation
                            if is_overflow or counts <= 1 or not self._is_initialized[j] or self._last_sync==0:
                                self._out_frequency.append(np.float64(np.nan))
                            else:
                                delta_c = float(self._current_counts[j] - self._last_counts[j])
                                delta_f = fractional_phase - self._last_fractional_phase[j]
                                delta_t_s = (self._current_sync - self._last_sync) * 1e-12
                                self._out_frequency.append(np.float64((delta_c + delta_f) / delta_t_s))

                            # save state for next window
                            self._last_counts[j] = self._current_counts[j]
                            self._current_counts[j] = self._cumulative_counts[j]
                            self._last_fractional_phase[j] = fractional_phase
                            self._is_initialized[j] = self._is_initialized[j] or (counts > 1 and not is_overflow)

                            # reset accumulators for next window
                            self._sum_delta_time[j] = 0
                            self._sum_i_delta_time[j] = 0
                            self.counts_per_channel[j] = 0
                        if self._current_sync != 0:
                            # sampling interval smaller than fitting window
                            if (tag['time'] - self._current_sync) <= self.fitting_window:
                                self._bad_window = True
                                break
                        self._last_sync = self._current_sync
                        self._current_sync = tag['time']
                        is_overflow = False
                else:
                    if self._current_sync != 0 and not is_overflow:
                        # relative time since window start
                        delta_time = tag["time"] - self._current_sync
                        for j in range(n_channels):
                            if self.channels[j] == tag['channel']:
                                # total cycle count increment
                                self._cumulative_counts[j] += 1
                                # increment accumulators only if inside fitting window
                                if 0 <= delta_time <= self.fitting_window:
                                    if self.counts_per_channel[j] == 0:
                                        # first in-window edge
                                        self._win_first_edge_time_ps[j] = tag["time"]
                                    # update last edge on every in-window event
                                    self._win_last_edge_time_ps[j] = tag["time"]
                                    self._sum_i_delta_time[j] += self._sum_delta_time[j]
                                    self._sum_delta_time[j] += delta_time
                                    self.counts_per_channel[j] += 1
                                break


    def __init__(self, tagger, channels, sync_channel,
                 fitting_window, max_windows=0):
        TimeTagger.CustomMeasurement.__init__(self,tagger)

        # Checks
        if not channels:
            raise ValueError("`channels` must contain at least one measurement channel.")
        if any((not isinstance(c, (int, np.integer))) for c in channels):
            raise TypeError("All entries in `channels` must be integers.")

        # no duplicates
        if len(set(channels)) != len(channels):
            duplicates = sorted({c for c in channels if channels.count(c) > 1})
            raise ValueError(f"Duplicate measurement channels not allowed: {duplicates}")

        # sync must be different from every measurement channel
        if sync_channel in channels:
            raise ValueError("`sync_channel` cannot also be in `channels`.")

        # channels to measure
        self.channels = np.array(channels, dtype=np.uint8)
        # Number of channels to measure
        self.n_channels = np.uint8(len(self.channels))
        # channel used to trigger windows
        self.sync_channel = np.uint8(sync_channel)
        # window length in ps
        self.fitting_window = int(fitting_window)
        # maximum stored windows (0=unlimited)
        self.max_windows = int(max_windows)

        # register both measurement and sync channels
        for ch in channels + [sync_channel]:
            self.register_channel(channel=ch)

        # create kernel before finalize_init
        self.kernel = self.Kernel(self.channels, self.sync_channel, self.fitting_window)

        self.finalize_init()


    def process(self, incoming_tags, begin_time, end_time):
        self.kernel.push(incoming_tags)




        if self.kernel._bad_window:
            self.kernel._bad_window = False
            raise ValueError(
            f"Fitting window ({self.fitting_window} ps) must be smaller than sync interval. "
            f"Reduce fitting window or slow down sync signal."
    )
        # enforce maximum stored windows
        if self.max_windows and len(self.kernel._sync_times[:-1]) > self.max_windows:
            drop = len(self.kernel._sync_times[:-1]) - self.max_windows
            for buf in (
                self.kernel._sync_times, self.kernel._counts, self.kernel._fractional_phase,
                self.kernel._phase, self.kernel._frequency, self.kernel._instantaneous_frequency,
                self.kernel._is_overflow
            ):
                del buf[:drop]

    def getDataObject(self):
        """
        Returns a FrequencyCounterData object
        """
        with self.mutex:
            n_sync, n_chan = len(self.kernel._out_sync_times[:-1]), len(self.channels)
            sync_times = np.array(self.kernel._out_sync_times, dtype=np.uint64)
            counts = np.array(self.kernel._out_counts, dtype=np.uint64).reshape(n_sync, n_chan)
            fractional_phase = np.array(self.kernel._out_fractional_phase, dtype=np.float64).reshape(n_sync, n_chan)
            phase = counts.astype(np.float64) + fractional_phase
            frequency = np.array(self.kernel._out_frequency, dtype=np.float64).reshape(n_sync, n_chan)
            instantaneous_frequency = np.array(self.kernel._out_instantaneous_frequency, dtype=np.float64).reshape(n_sync, n_chan)
            is_overflow = np.array(self.kernel._out_is_overflow, dtype=np.uint8).reshape(n_sync, n_chan)
            return FrequencyCounterData(sync_times, counts, fractional_phase, phase, frequency, instantaneous_frequency, is_overflow)


class FrequencyCounterData:
    """
    Data container for SyncFrequencyCounter.getDataObject().
    Provides getters for each dataset.
    """
    def __init__(self, times, counts, fraction, phase, frequency, inst_freq, overflow):
        self._times = times
        self._counts = counts
        self._fraction = fraction
        self._phase = phase
        self._frequency = frequency
        self._inst_freq = inst_freq
        self._overflow = overflow

    def getIndex(self):
        return np.arange(len(self._counts[:,0]))

    def getTime(self):
        return self._times[:-1]

    def getPeriodsCount(self):
        return self._counts.T

    def getPeriodsFraction(self):
        return self._fraction.T

    def getPhase(self):
        return self._phase.T

    def getFrequency(self):
        return self._frequency.T

    def getFrequencyInstantaneous(self):
        return self._inst_freq.T

    def getOverflowMask(self):
        return self._overflow.T

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    SIGNAL = 1
    SYNC = 2

    tagger = TimeTagger.createTimeTagger()
    [tagger.setTestSignal(ch, True) for ch in [SIGNAL, SYNC]]
    tagger.setEventDivider(SYNC, 937)

    # In this example the sync is periodic
    # Measure the rate of the Sync channel
    cr = TimeTagger.Countrate(tagger=tagger, channels=[SYNC])
    cr.startFor(1e12)
    cr.waitUntilFinished()
    rate = cr.getData()[0]

    # Compare the FrequencyCounter class with the custom SyncFrequencyCounter
    sm = TimeTagger.SynchronizedMeasurements(tagger=tagger)
    tagger_proxy = sm.getTagger()

    # Use as sampling interval for the FrequencyCounter the period of the sync
    sampling = 1e12//rate
    fc= TimeTagger.FrequencyCounter(tagger=tagger_proxy, channels=[SIGNAL], sampling_interval=sampling, fitting_window=sampling//2)
    sync_fc = SyncFrequencyCounter(tagger=tagger_proxy, channels=[SIGNAL], sync_channel=SYNC, fitting_window=sampling//2)

    DURATION = 20e12
    sm.startFor(DURATION)

    # Plot the instantaneous and averaged frequencies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    while sm.isRunning():
        plt.pause(.2)
        ax1.cla(); ax2.cla()
        fc_data = fc.getDataObject()
        sync_fc_data = sync_fc.getDataObject()

        sampling_times = fc_data.getTime()
        sync_times = sync_fc_data.getTime()

        ax1.plot((sync_times-sync_times[0])/1e12, sync_fc_data.getFrequencyInstantaneous()[0], marker='o', linewidth=0, color='k', label='CUSTOM')
        ax1.plot(sampling_times/1e12, fc_data.getFrequencyInstantaneous()[0],  marker='o', linewidth=0, label='API')
        ax1.set_ylabel('Instantaneous Frequency [Hz]')
        ax1.set_xlabel('Time [s]')
        ax1.grid(True)
        ax1.legend(loc="upper right")
        ax1.grid(which='minor', axis='x', linestyle=':')

        ax2.plot((sync_times-sync_times[0])/1e12, sync_fc_data.getFrequency()[0], marker='o', linewidth=0, color='k', label='CUSTOM')
        ax2.plot(sampling_times/1e12, fc_data.getFrequency()[0], marker='o', linewidth=0, label='API')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [s]')
        ax2.grid(True)
        ax2.legend(loc="upper right")
        ax2.grid(which='minor', axis='x', linestyle=':')

        fig.tight_layout()