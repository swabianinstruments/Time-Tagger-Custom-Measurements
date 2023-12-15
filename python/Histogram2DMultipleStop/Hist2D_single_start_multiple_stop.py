import matplotlib.pyplot as plt
import TimeTagger
import numpy as np
import numba


class Histogram2D_MultipleStop(TimeTagger.CustomMeasurement):
    """
    This is a code for a single start - multiple stop Histogram2D measurement.
    """

    def __init__(self, tagger, start_channel, stop_channel_1, stop_channel_2,
                 binwidth_1, binwidth_2, n_bins_1, n_bins_2, max_clicks):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.start_channel = start_channel
        self.stop_channel_1 = stop_channel_1
        self.stop_channel_2 = stop_channel_2
        self.binwidth_1 = binwidth_1
        self.binwidth_2 = binwidth_2
        self.max_bins_1 = n_bins_1
        self.max_bins_2 = n_bins_2
        self.max_clicks = max_clicks

        # The method register_channel(channel) activates
        # that data from the respective channels is transferred
        # from the Time Tagger to the PC.
        self.register_channel(channel=start_channel)
        self.register_channel(channel=stop_channel_1)
        self.register_channel(channel=stop_channel_2)

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
        with self.mutex:
            return self.data.copy()

    def getIndex_1(self):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        arr = np.arange(0, self.max_bins_1) * self.binwidth_1
        return arr

    def getIndex_2(self):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        arr = np.arange(0, self.max_bins_2) * self.binwidth_2
        return arr

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.last_start_timestamp = 0
        self.data = np.zeros(
            (self.max_bins_1, self.max_bins_2), dtype=np.uint64)

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
            stop_channel_1,
            stop_channel_2,
            binwidth_1,
            binwidth_2,
            max_clicks,
            last_start_timestamp):
        """
        A precompiled version of the histogram algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operation will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code
        """

        # The indices of where to sort the events will be stored here.
        # The maximum number of combinations you expect has to fit in.
        ind_x = np.zeros((max_clicks,), dtype=np.uint32)
        ind_y = np.zeros((max_clicks,), dtype=np.uint32)

        # Array for the timetags between the trigger signals to make use of vectorization
        clicks_x = np.zeros((max_clicks,), dtype=np.uint64)
        clicks_y = np.zeros((max_clicks,), dtype=np.uint64)

        # Pointer to the last stored tags
        last_x = 0
        last_y = 0
        start = False

        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents
            if tag['type'] != 0:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                last_start_timestamp = 0
            elif tag['channel'] == start_channel:
                if start is False:
                    last_start_timestamp = tag['time']
                    start = True
                # From the second trigger event onwards, we evaluate and sort
                # the time tags into the histogram
                else:
                    ind_x[0:last_x] = (clicks_x[0:last_x] -
                                       last_start_timestamp) // binwidth_1
                    ind_y[0:last_y] = (clicks_y[0:last_y] -
                                       last_start_timestamp) // binwidth_2
                    for x in ind_x[0:last_x]:
                        for y in ind_y[0:last_y]:
                            if x < data.shape[0] and y < data.shape[1]:
                                data[x, y] += 1
                last_start_timestamp = tag['time']
                last_x = 0
                last_y = 0
            # Storing the time tags in between trigger events
            # to make use of vectorization
            elif tag['channel'] == stop_channel_1 and start is True and last_start_timestamp != 0:
                clicks_x[last_x] = tag['time']
                last_x += 1
            elif tag['channel'] == stop_channel_2 and start is True and last_start_timestamp != 0:
                clicks_y[last_y] = tag['time']
                last_y += 1

        return last_start_timestamp

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
            not only the onces from register_channel(...).
        begin_time
            Begin timestamp of the of the current data block.
        end_time
            End timestamp of the of the current data block.
        """
        self.last_start_timestamp = Histogram2D_MultipleStop.fast_process(
            incoming_tags,
            self.data,
            self.start_channel,
            self.stop_channel_1,
            self.stop_channel_2,
            self.binwidth_1,
            self.binwidth_2,
            self.max_clicks,
            self.last_start_timestamp)


# Making sure the measure function is not executed at import
if __name__ == '__main__':
    click_ch_1 = 2
    click_ch_2 = 3
    start_ch = 1
    binwidth_x = 2500
    binwidth_y = 2500
    bins_x = 501
    bins_y = 501
    max_multiple_stops = 16
    meas_time = 2e12

    tagger = TimeTagger.createTimeTagger()
    model = tagger.getConfiguration()['devices'][0]['model']
    if model == 'Time Tagger 20':
        tagger.setTestSignalDivider(20)
    elif model == 'Time Tagger Ultra':
        tagger.setTestSignalDivider(16)
    elif model == 'Time Tagger X':
        tagger.setTestSignalDivider(100)
    else:
        raise ValueError('Model not yet supported with this script')
    tagger.setTestSignal([1, 2, 3], True)
    tagger.setEventDivider(start_ch, 10)
    sync_meas = TimeTagger.SynchronizedMeasurements(tagger)
    sync_tagger = sync_meas.getTagger()
    hist2d = Histogram2D_MultipleStop(
        sync_tagger, start_ch, click_ch_1, click_ch_2,
        binwidth_x, binwidth_y, bins_x, bins_y, max_multiple_stops
    )
    cnr = TimeTagger.Countrate(tagger, [start_ch, click_ch_1, click_ch_2])
    sync_meas.startFor(meas_time)
    sync_meas.waitUntilFinished()
    ind1 = hist2d.getIndex_1()
    ind2 = hist2d.getIndex_2()
    data = hist2d.getData()
    TimeTagger.freeTimeTagger(tagger)
    del tagger
    print(f'count rates: {cnr.getData()/1e6} MHz')
    plt.pcolormesh(ind1/1e6, ind2/1e6, data/1e3, cmap='magma_r')
    plt.xlabel('time difference (microseconds)')
    plt.ylabel('time difference (microseconds)')
    plt.colorbar(label='kcounts')
    plt.show()
