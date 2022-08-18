"""
EventSampler samples the events from input channels over a time window before a trigger event.

Author: Igor Shavrin, <igor@swabianinstruments.com>
License: BSD 3-Clause

This is similar to TimeTagger.Sampler but  instead of the expected signal level it 
samples if an event was present at a time window preceding the trigger event.

Event analysis diagram:
---o-----------------o---    Channel 1  [bit 0]
--o---------o--------o---    Channel 2  [bit 1]
----o----------o------o--    Channel 3  [bit 2]
----T--------T--------T---   Trigger channel
|...|----|...|----|...|      Time window before each trigger (sample time)
[t1,     [t2,      [t3,      Data [trigger_time,
b011]     b010]     b011]          bits]
========================>    Time axis

Usage example:
    es = EventSampler(
        tagger,             # TimeTagger object
        channels,           # Channel on which to sample the events
        trigger_channel,    # Trigger channel
        sample_window,      # Sample time window around the trigger to detect events
        max_samples         # Maximum number of samples in the buffer
    )

Dependencies: 
    numba
    numpy
    TimeTagger
"""

import TimeTagger
import numpy as np


class EventSampler(TimeTagger.CustomMeasurement):

    def __init__(self, tagger, channels, trigger_channel, sample_window, max_samples):

        if len(channels) > 64:
            raise ValueError('Maximum 64 channels are supported in a single instance.')

        super().__init__(tagger)
        self.channels = np.array(channels, dtype=np.int32)
        self.trigger_channel = trigger_channel
        self.sample_window = sample_window
        self.max_samples = max_samples

        self.current_index = 0
        self.data = np.zeros((self.max_samples, 2), dtype=np.int64)
        self.latest_timestamps = np.zeros((len(self.channels),), dtype=np.int64)

        # The method register_channel(channel) activates
        # data transfer from the respective hardware channel to the PC.
        for channel in self.channels:
            self.register_channel(channel)
        self.register_channel(self.trigger_channel)
        
        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.current_index = 0
        self.data.fill(0)
        self.latest_timestamps.fill(0)

    def getData(self):
        """ Return buffer contents and clear the buffer.
            The buffer is a 2D array of shape [max_samples, 2] and the contents are
            [[trigger_timestamp, bits_as_int], .... ].
        """
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        with self.mutex:
            ret_data = self.data[:self.current_index, :].copy()
            self.current_index = 0
            self.data.fill(0)
            return ret_data

    def process(self, incoming_tags, begin_time, end_time):
        """
        Main processing method for the incoming raw time-tags.

        The lock is already acquired within the backend.
        self.data is provided as reference, so it must not be accessed
        anywhere else without locking the mutex.

        Parameters
        ----------
        incoming_tags
            The incoming raw time tag stream provided as a read-only reference.
            The storage will be deallocated after this call, so you must not store a reference to
            this object. Make a copy instead.
            Please note that the time tag stream of all channels is passed to the process method,
            not only the ones from register_channel(...).
        begin_time
            Begin timestamp of the of the current data block.
        end_time
            End timestamp of the of the current data block.
        """
        
        # Add your processing code here. Consider making your code fast with numba JIT.
        self.current_index = fast_process(
            incoming_tags, 
            self.channels,
            self.trigger_channel,
            self.sample_window,
            self.data,           # a numpy array that can be modified in-place
            self.current_index,  # Current sample index
            self.latest_timestamps    # a numpy array that can be modified in-place
        ) 

    
import numba

@numba.jit(nopython=True, nogil=True)
def fast_process(tags, channels, trigger_channel, t_window, data, current_index, latest_timestamps):
    """
    Numba will precompile this function on-the-fly for better performance.
    Read more on the supported Python features:
        https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
    nopython=True: Only a subset of the python syntax is supported.
                    Avoid everything but primitives and numpy arrays.
                    All slow operation will yield an exception
    nogil=True:    This method will release the global interpreter lock. So
                    this method can run in parallel with other python code
    """

    channels_n = len(channels)

    for tag in tags:
        # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
        # OverflowEnd, 4 - MissedEvents
        if tag['type'] != 0:
            # tag is not a TimeTag, so we are in an error state, e.g. overflow
            continue

        if current_index >= data.shape[0]:
            # Buffer is full, skip this data until buffer is cleared.
            continue
            
        tag_time = tag["time"]
        tag_channel = tag['channel']

        if tag_channel != trigger_channel:
            # Collect latest timestamp for each channel
            for i in range(channels_n):
                if tag_channel == channels[i]:
                    latest_timestamps[i] = tag_time
                    break
        else:
            # This is a trigger event. Here we analyze timestamps within the 
            # sample window and set corresponding bits in a value.
            trigger_time = tag_time
            word = np.int64(0)

            for i in range(channels_n):
                if trigger_time - latest_timestamps[i] < t_window:
                    word |= 1 << i

            data[current_index, 0] = trigger_time
            data[current_index, 1] = word
            current_index += 1

    return current_index


if __name__ == '__main__':
    # DEMO SCRIPT

    import time
    CHANNELS = [2,3,4]
    TRIGGER_CH = 1

    tagger: TimeTagger.TimeTagger = TimeTagger.createTimeTagger()

    # enable the test signal
    tagger.setTestSignal(TRIGGER_CH, True)
    tagger.setInputDelay(TRIGGER_CH, 1000)
    for ch in CHANNELS:
        tagger.setTestSignal(ch, True)

    tagger.setEventDivider(TRIGGER_CH, 1)
    for ch, div in zip(CHANNELS, [2,4,8]):
        tagger.setEventDivider(ch, div)

    sampler = EventSampler(tagger, CHANNELS, TRIGGER_CH, sample_window=2000, max_samples=10000)

    for i in range(10):
        time.sleep(1)
        print(i, sampler.getData())

    """
    ATTENTION: This example is self contained and uses internal test signal divided by EventDivider. 
    Time Tagger's EventDivider works on each channel independently and therefore,
    the division start may not align at all channels. The results of several script executions 
    may differ from the expected value and may differ between runs. Since this is a limitation of the demo script only, 
    your actual measurements will return correct data.
    """