import matplotlib.pyplot as plt
import TimeTagger
import numpy as np
import numba
from time import sleep


class CountBetweenMarkersRolling(TimeTagger.CustomMeasurement):
    """
    CountBetweenMarkersRolling, authored by Fabian July 2022:
    This custom measurement is a CountBetweenMarkers measurement with a circular buffer and a rolling readout. 
    It works in a similar fashion to the standard Counter measurement and allows to run the measurement indefinitely with a continuous rolling readout. 
    Once the data array is full, it cycles back to beginning and starts writing over the old values. 
    This way, only the latest data is stored in the data array. 
    When quering the data, the user can chose whether a rolling or a sweeping readout is wanted with getData(rolling=True/False).
    Warning: This measurement does not implement a stop channel, it does not have proper handling of overflows (only throws a text message)
    The measurement was written in response to support case #841, where a customer was doing measurements on a quantum repeater node. 
    He was doing repeated CountBetweenMarkers measurements, but also wanted a continuous version, so that he could monitor the operation of his network node 24/7.
    """

    def __init__(self, tagger, click_channel, start_channel, n_values):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        assert n_values > 0, 'The bin number should be a positive integer value.'

        self.click_channel = click_channel
        self.start_channel = start_channel
        self.max_bins = int(n_values)

        # The method register_channel(channel) activates
        # that data from the respective channels is transferred
        # from the Time Tagger to the PC.
        self.register_channel(channel=click_channel)
        self.register_channel(channel=start_channel)

        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def __getData(self, rolling=True):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        if rolling:
            return np.append(
                self.data[self.index:].copy(), self.data[:self.index].copy())
        else:
            return self.data.copy()

    def __getIndex(self, rolling=True):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        if rolling:
                tmp = np.append(self.timestamps[self.index:].copy(
                ) - self.gaps, self.timestamps[:self.index].copy())
                return tmp.copy() - tmp[0]
        else:
                return self.timestamps.copy()

    def getIndex(self, rolling=True):
        with self.mutex:
            return self.__getIndex(rolling)
        
    def getData(self, rolling=True):
        with self.mutex:
            return self.__getData(rolling)

    def getIndexAndData(self, rolling=True):
        with self.mutex:
            return self.__getIndex(rolling), self.__getData(rolling)

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.index = 0
        self.first_timestamp = 0
        self.next_timestamp = 0
        self.gaps = 0
        self.timestamps = np.zeros((self.max_bins,), dtype=np.int64)
        self.data = np.zeros((self.max_bins,), dtype=np.uint64)
        self.tempory_bin_counter = 0

    def on_start(self):
        # The lock is already acquired within the backend.
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        pass

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def fast_process(
            tags,
            data,
            click_channel,
            start_channel,
            first_timestamp,
            next_timestamp,
            tempory_bin_counter,
            gap,
            timestamps,
            index):
        """
        A precompiled version of the histogram algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operation will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code
        """
        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType
            # IntEnum)
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                # throw a text warning
                print('overflows present, output may be inconsistent')
            if tag['channel'] == click_channel and first_timestamp != 0:
                # add a count to temporay bin counter
                tempory_bin_counter += 1
            if tag['channel'] == start_channel:
                # on marker signal, store fully integrated data in the current
                # bin and save the next marker time

                # check if this is the first start tag of the measurement
                if first_timestamp == 0:
                    first_timestamp = tag['time']

                if index == data.shape[0] - 1:
                    # if circular buffer is at the end, start a new time axis,
                    # save difference between last of old axis entry and first
                    # entry of new axis
                    gap = tag['time'] - first_timestamp
                    first_timestamp = tag['time']
                data[index] = tempory_bin_counter
                timestamps[index] = next_timestamp
                next_timestamp = tag['time'] - first_timestamp

                # move to next cell of the circular buffer
                # jump to beginning when at the end
                tempory_bin_counter = 0
                index = (index + 1) if index + 1 < data.shape[0] else 0

        return index, gap, first_timestamp, tempory_bin_counter, next_timestamp

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
        self.index, self.gaps, self.first_timestamp, self.tempory_bin_counter, self.next_timestamp = CountBetweenMarkersRolling.fast_process(
            incoming_tags,
            self.data,
            self.click_channel,
            self.start_channel,
            self.first_timestamp,
            self.next_timestamp,
            self.tempory_bin_counter,
            self.gaps,
            self.timestamps,
            self.index)


# Channel definitions
CHAN_START = 2
CHAN_STOP = 1

if __name__ == '__main__':

    print("""Rolling CountBetweenMarkers

Implementation of a custom CountBetweenMarkers measurement with a circular buffer and a rolling readout. Both the rolling and the sweeping readout will be shown in the example below.
""")
    # example uses the Time Tagger Virtual, also runs on TTU by initiating 'tagger' with the following command
    #tagger = TimeTagger.createTimeTagger()
    tagger = TimeTagger.createTimeTaggerVirtual()

    # enable the test signal
    tagger.setTestSignal([CHAN_START, CHAN_STOP], True)
    # delay the stop channel by 2 ns to make sure it is later than the start
    tagger.setInputDelay(CHAN_STOP, 2000)
    tagger.setReplaySpeed(1)

    BINS = 104

    # We first have to create a SynchronizedMeasurements object to synchronize
    # several measurements
    with TimeTagger.SynchronizedMeasurements(tagger) as measurementGroup:
        # Instead of a real Time Tagger, we initialize the measurement with the proxy object measurementGroup.getTagger()
        # This adds the measurement to the measurementGroup. In contrast to a normal initialization of a measurement, the
        # measurement does not start immediately but waits for an explicit
        # .start() or .startFor().
        custom_counter = CountBetweenMarkersRolling(
            measurementGroup.getTagger(), CHAN_STOP, CHAN_START, n_values=BINS)

        print("Acquire data...\n")
        measurementGroup.start()
        sleep(.6)

        index1, data1 = custom_counter.getIndexAndData(rolling=True)
        
        sleep(1.1)
        index2, data2 = custom_counter.getIndexAndData(rolling=True)



    TimeTagger.freeTimeTagger(tagger)

    plt.plot(index1, data1, label='output1')
    plt.plot(index2, data2, '--', label='output2')
    plt.xlabel('Time difference (ps)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
