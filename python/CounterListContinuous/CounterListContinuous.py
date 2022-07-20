import TimeTagger
import numpy as np
import numba

# CounterListContinuous generates a list of traces as the Counter class.
# Each trace is defined by an event of a start and stop channel.
# The acquisition for a trace stops when a stop event is received or
# binwidth*n_bins is exceeded.
# The class is optimized for continuous operations such that every
# .getData() call returns the acquired data and removes it.
#
#
# .getData(..., n_bins = 4, ...)
# [[12  2  3 12]  # trace one from first start and stop
#  [15 12 10  5]  # trace two from second start and stop
#  ....         ]


class CounterListContinuous(TimeTagger.CustomMeasurement):

    def __init__(self, tagger, start_channel, stop_channel, click_channel, binwidth, n_bins, cycle_buffer_size):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.start_channel = start_channel
        self.stop_channel = stop_channel
        self.click_channel = click_channel
        self.binwidth = binwidth
        self.max_bins = n_bins
        # maximum number of start/stop pairs before a .getData() call must be invoked
        self.cycle_buffer_size = cycle_buffer_size

        # The method register_channel(channel) activates
        # that data from the respective channels is transferred
        # from the Time Tagger to the PC.
        self.register_channel(channel=start_channel)
        self.register_channel(channel=stop_channel)
        self.register_channel(channel=click_channel)

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
        # This ensures a return of consistent data.
        with self.mutex:
            completed_slices = self.writeIndex

            if completed_slices == 0:
                return np.zeros((0, self.max_bins,), dtype=np.uint64)

            return_array = self.data[:completed_slices, :].copy()

            if self.last_start_timestamp == 0:
                # we are currently not in acquisition state so wipe all data
                self.data[:completed_slices, :] = 0
            else:
                # we are currently in acquisition state
                # copy the current data slice to index 0
                # wipe all other data
                self.data[0, :] = self.data[self.writeIndex, :]
                self.data[1:self.writeIndex+1, :] = 0
            self.writeIndex = 0
            return return_array

    def getIndex(self):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        arr = np.arange(0, self.max_bins, dtype=np.uint64) * self.binwidth
        return arr

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.last_start_timestamp = 0
        self.writeIndex = 0
        self.data = np.zeros((self.cycle_buffer_size, self.max_bins,), dtype=np.uint64)

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
            start_channel,
            stop_channel,
            click_channel,
            binwidth,
            last_start_timestamp,
            write_index):
        """
        A precompiled version of the algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operations will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code
        """
        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType IntEnum)
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # tag is not a TimeTag, so we are in an error state, e.g., overflow
                assert False, "Overflows are not handled yet!"
            elif tag['channel'] == click_channel and last_start_timestamp != 0:
                # valid event
                index = int((tag['time'] - last_start_timestamp) // binwidth)
                if index < data.shape[1]:
                    data[write_index, index] += 1
            elif tag['channel'] == start_channel:
                if last_start_timestamp != 0:
                    assert False, "There have been two start signals without a stop signal in between!"
                    # in case we allow for two start signals without a stop signal the following code can be used
                    write_index = write_index + 1
                    if write_index == data.shape[0]:
                        assert False, "Cycle buffer size exceeded!"
                last_start_timestamp = tag['time']
            elif tag['channel'] == stop_channel and last_start_timestamp != 0:
                write_index = write_index + 1
                if write_index == data.shape[0]:
                    assert False, "Cycle buffer size exceeded!"
                last_start_timestamp = 0
        return last_start_timestamp, write_index

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
        self.last_start_timestamp, self.writeIndex = CounterListContinuous.fast_process(
            incoming_tags,
            self.data,
            self.start_channel,
            self.stop_channel,
            self.click_channel,
            self.binwidth,
            self.last_start_timestamp,
            self.writeIndex)


def test_with_testsignal():
    tagger = TimeTagger.createTimeTaggerVirtual()
    tagger.setReplaySpeed(1)

    CHAN_START = 1
    CHAN_STOP = 2
    CHAN_CLICK = 3

    # enable the test signal and set the delays
    tagger.setTestSignal([CHAN_START, CHAN_STOP, CHAN_CLICK], True)
    tagger.setInputDelay(CHAN_CLICK, 1500)
    tagger.setInputDelay(CHAN_STOP, 3000)

    BINWIDTH = 1000  # ps
    BINS = 10

    counter = CounterListContinuous(tagger, CHAN_START, CHAN_STOP, CHAN_CLICK, binwidth=BINWIDTH, n_bins=BINS, cycle_buffer_size=1000)

    counter.startFor(int(1e7))
    counter.waitUntilFinished()

    x = counter.getIndex()
    ys = counter.getData()
    print(ys)


def test_with_data_from_file():
    tagger = TimeTagger.createTimeTaggerVirtual()

    CHAN_START = 3  # 1Hz
    CHAN_STOP = -3  # 50 duty cycle
    CHAN_CLICK = 1  # 100 kHz
    CHAN_CLICK2 = 2  # 10 kHz

    BINS = int(10)
    BINWIDTH = int(1e12/BINS)  # ps
    filename = 'CounterListContinuous.ttbin'

    sm = TimeTagger.SynchronizedMeasurements(tagger)
    countrate = TimeTagger.Countrate(sm.getTagger(), [CHAN_START, CHAN_STOP, CHAN_CLICK, CHAN_CLICK2])
    counter = CounterListContinuous(sm.getTagger(), CHAN_START, CHAN_STOP, CHAN_CLICK, binwidth=BINWIDTH, n_bins=BINS, cycle_buffer_size=1000)
    counter2 = CounterListContinuous(sm.getTagger(), CHAN_START, CHAN_STOP, CHAN_CLICK2, binwidth=BINWIDTH, n_bins=BINS, cycle_buffer_size=1000)
    sm.start()

    tagger.replay(filename)
    tagger.waitForCompletion()

    x = counter.getIndex()
    ys = counter.getData()
    ys2 = counter2.getData()

    total_counts = countrate.getCountsTotal()
    print("Total Counts of the events in the stream")
    print("Start  (CH {:2d}): {}".format(CHAN_START, total_counts[0]))
    print("Stop   (CH {:2d}): {}".format(CHAN_STOP, total_counts[1]))
    print("Click  (CH {:2d}): {}".format(CHAN_CLICK, total_counts[2]))
    print("Click2 (CH {:2d}): {}".format(CHAN_CLICK2, total_counts[3]))
    print()
    print("x")
    print(x)
    print("channel {}".format(CHAN_CLICK))
    print(ys)
    print("channel {}".format(CHAN_CLICK2))
    print(ys2)


def test_with_real_device():
    tagger = TimeTagger.createTimeTagger()

    CHAN_START = 3  # 1Hz
    CHAN_STOP = -3  # 50 duty cycle
    CHAN_CLICK = 1  # 100 kHz
    CHAN_CLICK2 = 2  # 10 kHz

    BINS = int(10)
    BINWIDTH = int(1e12/BINS)  # ps
    sm = TimeTagger.SynchronizedMeasurements(tagger)
    filename = 'CounterListContinuous.ttbin'
    counter = CounterListContinuous(sm.getTagger(), CHAN_START, CHAN_STOP, CHAN_CLICK, binwidth=BINWIDTH, n_bins=BINS, cycle_buffer_size=1000)
    counter2 = CounterListContinuous(sm.getTagger(), CHAN_START, CHAN_STOP, CHAN_CLICK2, binwidth=BINWIDTH, n_bins=BINS, cycle_buffer_size=1000)
    fw = TimeTagger.FileWriter(sm.getTagger(), filename, [CHAN_START, CHAN_STOP, CHAN_CLICK, CHAN_CLICK2])

    sm.startFor(3e12)
    sm.waitUntilFinished()

    x = counter.getIndex()
    ys = counter.getData()
    ys2 = counter2.getData()

    print("x")
    print(x)
    print("channel {}".format(CHAN_CLICK))
    print(ys)
    print("channel {}".format(CHAN_CLICK2))
    print(ys2)


if __name__ == '__main__':
    print('CustomCounter')
    # test_with_data_from_file()
    # test_with_real_device()
    test_with_testsignal()
