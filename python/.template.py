"""
Simple template for a Python implementation of a CustomMeasurement.
Author: Igor Shavrin, <igor@swabianinstruments.com>
License: BSD 3-Clause

This measurement counts time-differences between consecutive events
and collects them into an array with element width of 1 ns.

Usage example:
    cm = MyCustomMeasurement(
        tagger,   # TimeTagger object
        channel,  # Channel on which to count the events
        max_bins  # Number of elements in the data array.
    )

Dependencies: 
    numba
    numpy
    TimeTagger
"""

import TimeTagger
import numpy as np


class MyCustomMeasurement(TimeTagger.CustomMeasurement):

    def __init__(self, tagger, channel, max_bins):
        super().__init__(self, tagger)
        
        self.channel = channel
        self.max_bins = max_bins

        # The method register_channel(channel) activates
        # data transfer from the respective hardware channel to the PC.
        self.register_channel(channel)
        
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
        self.last_timestamp = 0
        self.data = np.zeros((self.max_bins,), dtype=np.uint64)

    def getData(self):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        with self.mutex:
            return self.data.copy()

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
        self.last_timestamp = fast_process(
            incoming_tags, 
            begin_time, 
            end_time, 
            self.channel,
            self.last_timestamp,  # immutable python object that shall be stored between calls of "fast_process"
            self.data             # a numpy array that can be modified in-place
        )

    
import numba

@numba.jit(nopython=True, nogil=True)
def fast_process(tags, begin_time, end_time, channel, last_timestamp, data):
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
    for tag in tags:
        # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
        # OverflowEnd, 4 - MissedEvents
        if tag['type'] != 0:
            # tag is not a TimeTag, so we are in an error state, e.g. overflow
            last_timestamp = 0
        elif tag['channel'] == channel and last_timestamp != 0:
            # valid event
            index = (tag['time'] - last_timestamp) // 1000
            if index < data.shape[0]:
                data[index] += 1
            last_timestamp = tag['time']
    return last_timestamp

    
if __name__ == '__main__':
    # Add your test code here

    CHANNEL = 1

    tagger = TimeTagger.createTimeTagger()

    # enable the test signal
    tagger.setTestSignal(CHANNEL, True)

    cm = MyCustomMeasurement(tagger, CHANNEL, 1000)
    cm.startFor(1e12)
    cm.waitUntilFinished()
    print(cm.getData())

