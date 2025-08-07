import matplotlib.pyplot as plt
import TimeTagger
import numpy as np
import numba


class ContinuousHistogramming(TimeTagger.CustomMeasurement):
    """
    Example for a continuous histogramming measurement with multiple histograms.

    This measurement creates multiple histograms and switches between them when
    a signal on the next_channel is received. When the maximum number of histograms
    is reached, integration stops until getData(remove=True) is called.
    The getData(remove=True) method can be used to retrieve and remove fully
    integrated histograms, allowing for continuous operation.

    The measurement starts after the first signal on next_channel is received.
    When the internal buffer is exceeded, integration will start again only when
    the data is removed AND the next signal on next_channel is received.
    """

    # Define the spec for the jitclass - must be at module level for Numba
    histogram_spec = [
        ('data', numba.types.Array(numba.uint64, 3, 'C')),
        ('click_channels', numba.types.Array(numba.int32, 1, 'C')),
        ('n_click_channels', numba.int32),
        ('start_channel', numba.int32),
        ('next_channel', numba.int32),
        ('binwidth', numba.int64),
        ('n_bins', numba.int64),
        ('n_histogram_sets', numba.int64),
        ('current_histogram_set', numba.int64),
        ('last_start_timestamp', numba.int64),
        # Number of histogram sets that have been fully completed since measurement start
        ('histogram_sets_completed', numba.int64),
        ('total_next', numba.int64),                # Total number of signals on next_channel
        ('warning_shown', numba.types.boolean),
        ('waiting_for_next_signal', numba.types.boolean),
        ('in_overflow', numba.types.boolean),
    ]

    # Define the HistogramProcessor as an inner class
    @numba.experimental.jitclass(histogram_spec)
    class HistogramProcessor:
        """
        JIT-compiled class for histogram processing.
        This inner class handles the performance-critical processing of time tags.
        """

        def __init__(self, click_channels, start_channel, next_channel, binwidth, n_bins, n_histogram_sets):
            """
            Initialize the HistogramProcessor.

            Parameters
            ----------
            click_channels : list of int
                List of channel numbers for the click events
            start_channel : int
                Channel number for the start events
            next_channel : int
                Channel number to trigger switching to the next histogram
            binwidth : int
                Width of the histogram bins in picoseconds
            n_bins : int
                Number of bins in each histogram
            n_histogram_sets : int
                Maximum number of histogram sets (each set contains one histogram per click channel)
            """
            self.click_channels = np.array(click_channels, dtype=np.int32)
            self.n_click_channels = len(self.click_channels)
            self.start_channel = start_channel
            self.next_channel = next_channel
            self.binwidth = binwidth
            self.n_bins = n_bins
            self.n_histogram_sets = n_histogram_sets
            self.data = np.zeros((n_histogram_sets, self.n_click_channels, n_bins), dtype=np.uint64)
            self.clear(init=True)

        def clear(self, init=False):
            """
            Reset the processor state.

            This method resets the data array, current histogram set index,
            and last start timestamp without recreating the processor.
            """
            if not init:
                self.data.fill(0)  # Reset data to zero, not required on initialization
            self.current_histogram_set = 0
            self.last_start_timestamp = 0
            self.total_next = 0
            self.in_overflow = False
            self.histogram_sets_completed = 0  # Reset completed histogram sets counter
            self.warning_shown = False  # Reset warning flag
            self.waiting_for_next_signal = True  # Wait for first next_channel signal before starting integration

        def getData(self, remove=False):
            # Calculate theoretical total number of histogram sets that would have been generated without overflow constraints
            # This is derived from total signals on next_channel minus 1 (since first signal starts but doesn't increment)
            histogram_sets_total = max(0, self.total_next - 1)

            if not remove:
                # For remove=False, return a copy of all data
                return self.data.copy(), self.histogram_sets_completed, histogram_sets_total
            else:
                # For remove=True:

                # If current_histogram_set is -1, the data buffer is completely filled
                exceeded_limit = self.current_histogram_set < 0

                # Determine number of completed histogram sets
                num_completed = self.n_histogram_sets if exceeded_limit else self.current_histogram_set

                if num_completed == 0:
                    # No completed histogram sets to remove, return empty array
                    return np.zeros((0, self.n_click_channels, self.n_bins), dtype=np.uint64), self.histogram_sets_completed, histogram_sets_total

                # Create a copy of only the completed histogram sets
                data_copy = self.data[:num_completed].copy()

                if exceeded_limit:
                    # There is no histogram set where data is currently accumulated into.
                    # Everything must be set to 0.
                    self.data.fill(0)
                else:
                    # copy the current histogram set to the first position
                    # and clear the rest of the data
                    self.data[0] = self.data[self.current_histogram_set]
                    self.data[1:self.current_histogram_set + 1] = 0

                # Reset state
                self.current_histogram_set = 0

                # Reset warning flag
                self.warning_shown = False

                # If we were at the limit, we need to wait for the next signal
                # to restart integration. This ensures integration only restarts
                # after both data removal AND the next next_channel signal.
                if exceeded_limit:
                    self.waiting_for_next_signal = True

                return data_copy, self.histogram_sets_completed, histogram_sets_total

        def process_tags(self, tags):
            """
            Process time tags and update histograms.

            Parameters
            ----------
            tags : numpy.ndarray
                Array of time tags

            Returns
            -------
            int
                Number of histogram sets completed during this processing
            """
            # If current_histogram_set is already -1, we've already exceeded the limit
            # in a previous call, so we should continue to ignore data
            exceeded_limit = self.current_histogram_set < 0

            for tag in tags:
                # Check for error states
                if tag['type'] != TimeTagger.TagType.TimeTag:
                    # Tag is not a TimeTag, so we are in an error state, e.g. overflow
                    if tag['channel'] == self.next_channel and tag['type'] == TimeTagger.TagType.MissedEvents:
                        # Handle missed events on the next channel
                        self.total_next += tag['missed_events']
                    if self.in_overflow:
                        continue  # Ignore overflow tags if already in overflow state
                    self.in_overflow = True  # Set overflow state
                    self.last_start_timestamp = 0  # Reset last start timestamp
                    if self.current_histogram_set >= 0:
                        self.data[self.current_histogram_set, :, :] = 0  # Clear current histogram set for all channels
                    continue
                else:
                    self.in_overflow = False

                # Handle next channel events
                if tag['channel'] == self.next_channel:
                    self.total_next += 1
                    if not exceeded_limit:
                        if self.waiting_for_next_signal:
                            # First next_channel signal starts integration but doesn't increment histogram
                            self.waiting_for_next_signal = False
                        else:
                            # Subsequent next_channel signals increment to the next histogram set
                            self.current_histogram_set += 1
                            self.histogram_sets_completed += 1

                        # Check if we've reached the maximum number of histogram sets
                        if self.current_histogram_set >= self.n_histogram_sets:
                            # signal that we've exceeded the limit
                            exceeded_limit = True
                            self.current_histogram_set = -1
                            self.waiting_for_next_signal = True

                # Handle start channel events
                elif tag['channel'] == self.start_channel:
                    self.last_start_timestamp = tag['time']

                # Handle click channel events
                elif self.last_start_timestamp != 0 and not self.waiting_for_next_signal and not exceeded_limit:
                    # Check if this tag is from one of our click channels
                    for ch_idx in range(self.n_click_channels):
                        if tag['channel'] == self.click_channels[ch_idx]:
                            # Only process click events if:
                            # 1. Integration is active (not waiting for next signal)
                            # 2. We have a valid start timestamp
                            # 3. We haven't exceeded the histogram set limit
                            # Calculate the bin index
                            index = (tag['time'] - self.last_start_timestamp) // self.binwidth

                            # Check if the index is within range and current_histogram_set is valid
                            if index < self.n_bins and self.current_histogram_set >= 0 and self.current_histogram_set < self.n_histogram_sets:
                                # Increment the bin count in the current histogram for this channel
                                self.data[self.current_histogram_set, ch_idx, index] += 1
                            # No break here to allow processing duplicate channels

            # Check if we've exceeded the maximum number of histogram sets
            if exceeded_limit and not self.warning_shown:
                # Show warning, but only once
                print("WARNING: Maximum number of histogram sets reached. Use getData(remove=True) and a high enough poll rate to remove fully integrated histogram sets.")
                self.warning_shown = True

    def __init__(self, tagger, click_channels, start_channel, next_channel, binwidth, n_bins, n_histogram_sets):
        """
        Initialize the ContinuousHistogramming measurement.

        Parameters
        ----------
        tagger : TimeTagger
            TimeTagger object
        click_channels : list of int
            List of channel numbers for the click events
        start_channel : int
            Channel number for the start events
        next_channel : int
            Channel number to trigger switching to the next histogram set
        binwidth : int
            Width of the histogram bins in picoseconds
        n_bins : int
            Number of bins in each histogram
        n_histogram_sets : int
            Maximum number of histogram sets (each set contains one histogram per click channel)
        """
        TimeTagger.CustomMeasurement.__init__(self, tagger)

        # Validate click_channels
        if len(click_channels) == 0:
            raise ValueError("At least one click channel must be provided")

        self.click_channels = click_channels
        self.start_channel = start_channel
        self.next_channel = next_channel
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.n_histogram_sets = n_histogram_sets

        # Register channels for data transmission
        for channel in self.click_channels:
            self.register_channel(channel=channel)
        self.register_channel(channel=start_channel)
        self.register_channel(channel=next_channel)

        # Initialize the processor once
        self.processor = self.HistogramProcessor(
            self.click_channels,
            self.start_channel,
            self.next_channel,
            self.binwidth,
            self.n_bins,
            self.n_histogram_sets
        )

        self.clear_impl()

        # Finalize initialization
        self.finalize_init()

    def __del__(self):
        # Stop the measurement before deconstruction to avoid
        # concurrent process() calls
        self.stop()

    def getData(self, remove=False):
        """
        Get the histogram data and the total number of completed histogram sets.

        Parameters
        ----------
        remove : bool, optional
            If True, return and remove fully integrated histogram sets.
            The current unfinished histogram set will be moved to index 0,
            allowing for continuous operation without hitting the n_histogram_sets limit.
            This also resets the warning flag and allows integration to continue
            if the limit was previously exceeded.

        Returns
        -------
        tuple
            (data, histogram_sets_completed, histogram_sets_total)
            data : numpy.ndarray
                Array containing the histogram data. If remove=True, only the completed
                histogram sets are returned. If remove=False, all histogram sets are returned.
            histogram_sets_completed : int
                Number of histogram sets that have been fully completed since measurement start
            histogram_sets_total : int
                Theoretical total number of histogram sets that would have been generated
                if there were no overflow conditions (from Time Tag stream or n_histogram_sets limit)
        """
        # Lock the instance to ensure process() is not running in parallel
        with self.mutex:
            return self.processor.getData(remove)

    def getIndex(self):
        """
        Get the time axis for the histograms.

        Returns
        -------
        numpy.ndarray
            Array containing the time axis
        """
        # This method does not depend on the internal state, so no lock needed
        return np.arange(0, self.n_bins) * self.binwidth

    def clear_impl(self):
        """
        Reset the measurement data.
        """
        # The lock is already acquired within the backend
        # Reset the processor state instead of creating a new one
        self.processor.clear()

    def on_start(self):
        """
        Called when the measurement is started.
        """
        # The lock is already acquired within the backend
        pass

    def on_stop(self):
        """
        Called when the measurement is stopped.
        """
        # The lock is already acquired within the backend
        pass

    def process(self, incoming_tags, begin_time, end_time):
        """
        Main processing method for the incoming raw time-tags.

        Parameters
        ----------
        incoming_tags : numpy.ndarray
            Array of incoming time tags
        begin_time : int
            Begin timestamp of the current data block
        end_time : int
            End timestamp of the current data block
        """
        # Process the incoming tags using the JIT-compiled processor
        self.processor.process_tags(incoming_tags)


# Example usage
if __name__ == '__main__':
    # Definitions
    CHAN_START = 1
    CHAN_STOP1 = 2
    CHAN_STOP2 = 3
    CHAN_NEXT = 4

    BINWIDTH = 1  # ps
    BINS = 1000
    N_HISTOGRAM_SETS = 30  # data buffer: maximum number of histogram sets

    print("""ContinuousHistogramming example

Implementation of a continuous histogramming measurement with multiple histograms.
The measurement switches to the next histogram set when a signal on the next_channel is received.
When the maximum number of histogram sets is reached, integration stops until getData(remove=True) is called.
""")

    tagger = TimeTagger.createTimeTagger()

    # Enable the test signal
    tagger.setTestSignal([CHAN_START, CHAN_STOP1, CHAN_STOP2, CHAN_NEXT], True)
    # Delay the stop channels to make sure they are later than the start
    tagger.setInputDelay(CHAN_STOP1, int(BINS / 2))  # Use same delay for all click channels
    tagger.setInputDelay(CHAN_STOP2, int(BINS / 2))
    # Set the event divider for the next channel to the maximum value resulting in CHAN_NEXT frequency of about 14 Hz
    tagger.setEventDivider(CHAN_NEXT, 65535)

    # Create the measurement with multiple click channels
    measurement = ContinuousHistogramming(
        tagger,
        [CHAN_STOP1, CHAN_STOP2],  # List of click channels
        CHAN_START,
        CHAN_NEXT,
        binwidth=BINWIDTH,
        n_bins=BINS,
        n_histogram_sets=N_HISTOGRAM_SETS
    )

    print("Acquire data and plot every 1 second...\n")
    measurement.start()

    # Plot data every 1 second
    plt.figure(figsize=(12, 8))
    plt.ion()  # Turn on interactive mode for dynamic updates

    overflow_set_no = 5
    for i in range(7):  # Run 10 times

        if i == overflow_set_no:
            print("INFO: Integration time is increased to 4 seconds to test the casse that N_HISTOGRAM_SETS are exceeded.")
        integration_time = 1 if i < overflow_set_no else 4
        plt.pause(integration_time)

        # Get the data and total completed histogram sets
        data, histogram_sets_completed, histogram_sets_total = measurement.getData(remove=True)

        # Clear the previous plot
        plt.clf()

        # If no data yet, continue
        if len(data) == 0:
            plt.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', fontsize=14)
            plt.draw()
            continue

        # Create subplots for each click channel
        n_channels = data.shape[1]
        for ch_idx in range(n_channels):
            plt.subplot(n_channels, 1, ch_idx+1)

            # Get time axis
            x_index = measurement.getIndex()

            # Plot histograms for this channel
            for j in range(len(data)):
                plt.plot(x_index, data[j, ch_idx],
                         label=f'Histogram Set {j}, Channel {measurement.click_channels[ch_idx]}')

            plt.xlabel('Time difference (ps)')
            plt.ylabel('Counts')
            plt.title(f'Channel {measurement.click_channels[ch_idx]}')

            # Only add legend to the first subplot to save space
            if ch_idx == 0:
                plt.legend()

        plt.tight_layout()
        plt.suptitle(f'Returned Histogram Sets at t={i+1}s', fontsize=16, y=1.02)
        plt.draw()

        print(
            f"Integration time : ~{integration_time}s, Returned histogram sets: {len(data)}, "
            f"Completed histogram sets: {histogram_sets_completed}, Total histogram sets (theoretical): {histogram_sets_total}")

    # Stop the measurement
    measurement.stop()

    # Free the tagger
    TimeTagger.freeTimeTagger(tagger)
