import TimeTagger
import numpy as np
import numba
from collections import deque
from time import sleep
from datetime import datetime


class TimeDifferenceList2ChanContinuous(TimeTagger.CustomMeasurement):
    """
    Custom Measurement for QKD with a time-bin protocol
    Internal note: ticket 438
    Author: Michael Schlagmüller
    Date: October 2021

    Sync signal (start) to one channel and the quantum signals to the other (click),
    so that the timestamps restart from zero every time there is a detection on the start channel.

    multiple start-multiple stop

    1 start channel, 2 stop channels

    Similar to TimeDifference but instead of a Histogram the time differences are returned.

    Example

    data = measurement.getData()

    Stop-clicks: 748408 # len(data)
    print(datA)
    # t of start (ps)    td to stop   channel number of stop signal
    [[1352061357154          1716             2]
     [1352062467282       1111844             3]
     [1352064681694       3326256             2]
     ...
     [2352059749989       1123944             2]
     [2352060867665       2241620             3]
     [2352063089739       4463694             2]]

    sleep(...)

    data = measurement.getData()
    Stop-clicks: 739853 # len(data)
    # t of start (ps)    td to stop   channel number of stop signal
    [[2465715809473       4505242             2]
     [2465715809574       4505343             3]
     [2465718065512       1127294             3]
     ...
     [3465711889335       4518677             2]
     [3465711889422       4518764             3]
     [3465714132032       1121347             3]]

    Parameters

    interval:     Data is returned as a list and every a list entry is completed when the 'interval' time has passed.
    buffer_size:  buffer size of the internal buffer used
    """

    def __init__(self, tagger, click_channel1, click_channel2, start_channel, interval, buffer_size):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        assert click_channel1 != start_channel
        assert click_channel2 != start_channel
        assert click_channel1 != click_channel2
        buffer_size = int(buffer_size)
        interval = int(interval)
        assert buffer_size > 0
        assert interval >= 1e11, "The interval should not bee to short, at least 0.1s == 1e11 ps"
        self.click_channel1 = click_channel1
        self.click_channel2 = click_channel2
        self.start_channel = start_channel
        self.interval = interval
        # we increase the buffer size by one, because we always leave one row empty
        buffer_size = buffer_size + 1
        self.buffer_size = buffer_size
        self.data = np.zeros([buffer_size, 3], dtype=np.int64)
        # we need a list for returning the added block boundaries
        # in case the hard coded buffer size of 10000 is exceeded, an exception is thrown
        self.interval_new_start_indices = np.zeros(10000, dtype=np.int32)

        # The method register_channel(channel) activates
        # that data from the respective channels is transferred
        # from the Time Tagger to the PC.
        self.register_channel(channel=click_channel1)
        self.register_channel(channel=click_channel2)
        self.register_channel(channel=start_channel)

        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def getData(self):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        self._lock()
        if (self.read_index == self.write_index):
            # we have not stored anything - return an empty array
            self._unlock()
            return np.zeros([0, 3], dtype=np.int64)
        else:
            # valid state - data is stored
            buffered_time = self.__getBufferedTime()
            if buffered_time < self.interval:
                # check integrity - should never fail
                if len(self.interval_boundary_list) > 0:
                    self._unlock()
                    assert False, "len(self.interval_boundary_list) == 0"
                print("Internal buffered time {:.2e} is less that the interval defined at construction time {:.2e}. Please check with has_new_data() whether data is ready to be returned via .getData()".format(
                    buffered_time, self.interval))
                self._unlock()
                return np.zeros([0, 3], dtype=np.int64)

            # check integrity - should never fail
            if len(self.interval_boundary_list) == 0:
                self._unlock()
                assert False, "len(self.interval_boundary_list) > 0"
            next_interval_start = self.interval_boundary_list.pop()
            if next_interval_start >= self.read_index:
                # no wrap read case
                arr = self.data[self.read_index:next_interval_start, :].copy()
            else:
                # wrap read case
                # concatenate does a copy - so no explicit copy required
                arr = np.concatenate((self.data[self.read_index:self.buffer_size, :], self.data[0:next_interval_start, :]))
            self.read_index = next_interval_start
            # We have gathered the data, unlock, so measuring can continue.
        self._unlock()
        return arr

    def getBufferedTime(self):
        self._lock()
        buffered_time = self.__getBufferedTime()
        self._unlock()
        return buffered_time

    def __getBufferedTime(self):
        if self.read_index == self.write_index:
            buffered_time = 0
        else:
            buffered_time = self.data[(self.write_index-1), 0] - (self.init_time + (self.interval_counter-len(self.interval_boundary_list)) * self.interval)
        return buffered_time

    def has_new_data(self):
        self._lock()
        if len(self.interval_boundary_list) > 0:
            if not (self.__getBufferedTime() >= self.interval):
                self._unlock()
                assert False, "bufferedTime {} must be greater or equal interval {}".format(self.__getBufferedTime(), self.interval)
            result = True
        else:
            if not (self.__getBufferedTime() < self.interval):
                self._unlock()
                assert False, "self.__getBufferedTime() {} < self.interval {}".format(self.__getBufferedTime(), self.interval)
            result = False
        self._unlock()
        return result

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.start_channel_timestamp = 0
        self.init_time = 0
        self.write_index = 0
        self.read_index = 0
        self.interval_boundary_list = deque()
        self.interval_counter = 0

    def on_start(self):
        # The lock is already acquired within the backend.
        self.clear_impl()

    def on_stop(self):
        # The lock is already acquired within the backend.
        pass

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def fast_process(
            tags,
            data,
            click_channel1,
            click_channel2,
            start_channel,
            start_channel_timestamp,
            init_time,
            write_index,
            read_index,
            interval,
            interval_counter,
            interval_new_start_indices,
    ):
        """
        A precompiled version of the algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operations will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code.
        """
        _buffer_size = len(data)
        new_start_indices_buffer_size = len(interval_new_start_indices)
        finished_intervals = 0
        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType IntEnum)
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                # we only do simply overflow-handling, do not use this method with overflows
                # do the same as clear_impl()
                print("Time Tagger hardware Buffer overflow - reset local buffer")
                return -1, -1, -1, -1, -1
            else:
                if init_time == 0:
                    # when we did not call get data before - take the current time as the last get data call
                    init_time = tag['time']

                if (tag['channel'] == click_channel1 or tag['channel'] == click_channel2) and start_channel_timestamp != 0:
                    # check whether the buffer is exceeded
                    if (write_index + 1 == read_index) or (write_index == _buffer_size - 1 and read_index == 0):
                        print("Internal buffer exceeded. Please use a bigger buffer or call .getData() faster.")
                        break
                    # write a new timestamp
                    data[write_index, :] = [tag['time'], tag['time'] - start_channel_timestamp, tag['channel']]
                    write_index = write_index + 1
                    if write_index == _buffer_size:
                        write_index = 0

                    # did we cross the interval boundary
                    while tag['time'] >= init_time + interval * (interval_counter + 1):
                        assert finished_intervals < new_start_indices_buffer_size, "Error: buffer for boundaries within fast_process exceeded."
                        interval_new_start_indices[finished_intervals] = write_index
                        interval_counter += 1
                        finished_intervals += 1

                elif tag['channel'] == start_channel:
                    # found a new start timestamp
                    start_channel_timestamp = tag['time']

        return start_channel_timestamp, init_time, write_index, read_index, interval_counter

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
            Begin timestamp of the current data block.
        end_time
            End timestamp of the current data block.
        """
        interval_counter_before = self.interval_counter
        self.start_channel_timestamp, self.init_time, self.write_index, self.read_index, self.interval_counter = TimeDifferenceList2ChanContinuous.fast_process(
            incoming_tags,
            self.data,
            self.click_channel1,
            self.click_channel2,
            self.start_channel,
            self.start_channel_timestamp,
            self.init_time,
            self.write_index,
            self.read_index,
            self.interval,
            self.interval_counter,
            self.interval_new_start_indices,
        )
        # -1: error state returned from fast_process - reset the measurement class
        if self.start_channel_timestamp == -1:
            self.clear_impl()
        else:
            new_intervals = self.interval_counter - interval_counter_before
            # add the new interval boundaries
            for i in range(new_intervals):
                self.interval_boundary_list.appendleft(self.interval_new_start_indices[i])


# Channel definitions
CHAN_START = 1
CHAN_STOP1 = 2
CHAN_STOP2 = 3

if __name__ == '__main__':

    print("""Custom Measurement example

Implementation of a custom single start, multiple stop measurement.

The custom implementation will return a numpy 2D array.
Each row contains 
1) absolute time stamp
2) relative time stamp to the last start timestamp
3) channel number
A row is created for each stop click on the stop channels after a first start click has been received.

With the construction parameter interval (time duration), the returned blocks from .getData() will be defined, e.g. 1e12 ps = 1s.
Whether a new data block is ready to be returned can be checked via has_new_data().
When this returns True, call .getData() to get the data of the given time interval and deletes it from the internal buffer.
There will be no gap between the returned blocks.
""")

    tagger = TimeTagger.createTimeTagger()

    test_signal = True

    buffer_size = 10e6  # number of stop events which can be buffered before the .getData() call

    interval = 1e12  # duration in ps for each returned block with .getData()

    tdl = TimeDifferenceList2ChanContinuous(tagger, CHAN_STOP1, CHAN_STOP2, CHAN_START, interval, buffer_size)

    if test_signal:
        # enable the test signal
        tagger.setTestSignal([CHAN_START, CHAN_STOP1, CHAN_STOP2], True)
        # we reduce the data rate of the start channel to have two stops for one start
        tagger.setEventDivider(CHAN_START, 5)
        tagger.setEventDivider(CHAN_STOP1, 3)
        tagger.setEventDivider(CHAN_STOP2, 2)
        # delay the stop channel by 2 ns to make sure it is later than the start
        tagger.setInputDelay(CHAN_STOP1, 2000)
        tagger.setInputDelay(CHAN_STOP2, 2001)

    print("Poll Time Difference List for 5 seconds")
    start = datetime.now()
    while(True):
        sleep(0.2)
        if tdl.has_new_data():
            data = tdl.getData()
            print("Stop-clicks: {}".format(len(data)))
            print(data)
        if (datetime.now() - start).total_seconds() > 5:
            break
    del tagger
