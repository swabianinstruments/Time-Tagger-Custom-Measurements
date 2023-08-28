"""
Counts coincidence events on groups of channels with support of exclusion channels.
Author: Igor Shavrin, <igor@swabianinstruments.com>
License: BSD 3-Clause

This measurement detects and counts coincidences and supports the definition of 
exclusion channels. 
You can define coincidences such that successful coincidence is when there 
are events on included channels but no events on any of the excluded channels.
This measurement is similar to a combination of Coincidences and Countrate.
https://www.swabianinstruments.com/static/documentation/TimeTagger/api/VirtualChannels.html#coincidences
https://www.swabianinstruments.com/static/documentation/TimeTagger/api/Measurements.html#countrate

Usage example:
    ecc = ExclusiveCoincidenceCounter(
        tagger,   # TimeTagger object
        coincidenceGroups,  # A list of coincidence channel groups, 
                            #   [
                            #       ([<include channels>], [<exclude channels>]),
                            #       ([1,2,3], []),  Coincidence of channels 1&2&3
                            #       ([1,2], [3]), Coincidence of channels 1&2 but must have no event on 3.
                            #   ]
        coincidenceWindow   # [ps] Time window for coincidence analysis
    )

Limitations:
    - Works only with real input channels (rising and falling edges)

Dependencies: 
    numba
    numpy
    TimeTagger
"""

from typing import Union, NamedTuple
import numba
import numpy as np
import TimeTagger


# Exclusive coincidence group
class Group(NamedTuple):
    incl: list[int]
    excl: list[int]

# Offset used to shift channel numbers so they are always positive 
# This works for TT20, TTU, TTX.
CHAN_OFFSET = 18


class ExclusiveCoincidenceCounter(TimeTagger.CustomMeasurement):
    def __init__(self,
                 tagger: TimeTagger.TimeTaggerBase,
                 coincidenceGroups: Union[list[Group], list[tuple[list[int], list[int]]]],
                    # [([included_channels], [excluded_channels]), ... ([1,3], [2])]
                 coincidenceWindow: int):
        super().__init__(tagger)
        self._coincidenceGroups = coincidenceGroups
        self._coincidenceWindow = coincidenceWindow
        self._ring_size = 2**8 # Number of time tags to store for coincidence calculation. MUST BE POWER OF 2
        
        self._coin_groups_n = len(self._coincidenceGroups)
        self._coin_group_bitmap = np.zeros((self._coin_groups_n,), dtype=np.int64)
        self._coin_group_mask = np.zeros((self._coin_groups_n,), dtype=np.int64)

        # Create coincidence group channel bitmaps and channel participation masks
        for group_i, (incl, excl) in enumerate(self._coincidenceGroups):
            # 1 for all channels that must be present in coincidence, 0 for channels that must be absent
            # This bitmap will identify coincidence condition for the given group
            self._coin_group_bitmap[group_i] = np.sum(
                np.power(
                    2, np.array(incl) + CHAN_OFFSET, 
                    dtype=np.int64
                )
            )
            # 1: for all channels we care about (both, included and excluded), 0: for channels we do not care about.
            # This mask is needed to group coincidences for combinations of don't care channels.
            self._coin_group_mask[group_i] = np.sum(
                np.power(
                    2, np.array(incl+excl) + CHAN_OFFSET, 
                    dtype=np.int64
                )
            )

        # Register all required channels
        # NOTE: Repeated calls to self.register_channel() for already registered channel have no effect.
        for incl, excl in self._coincidenceGroups:
            for ch in excl+incl:
                self.register_channel(ch)

        self.clear_impl()
        self.finalize_init()

    def __del__(self):
        self.stop()

    def clear_impl(self):
        # Create ring buffer
        self._ring_buf = np.zeros(
            (self._ring_size,), 
            dtype=TimeTagger.CustomMeasurement.INCOMING_TAGS_DTYPE
        )
        self._ring_head = 0
        self._ring_tail = 0

        # Data array that contains counts of all possible coincidences.
        self.coin_counter = np.zeros((self._coin_groups_n,), dtype=np.int32)

    def getData(self):
        with self.mutex:
            # This code is blocking the processing thread and shall be as quick as possible.
            # We simply copy the data array.
            data = self.coin_counter.copy()
        return data

    def process(self, incoming_tags, begin_time, end_time):
        self._ring_head, self._ring_tail = fast_process(
            incoming_tags, 
            self.coin_counter, 
            CHAN_OFFSET,
            self._coin_group_bitmap,
            self._coin_group_mask,
            self._ring_buf, self._ring_head, self._ring_tail,
            self._coincidenceWindow,
        )


@numba.jit(nopython=True, nogil=True)
def fast_process(tags, counter, chan_offset, g_bitmap, g_mask, ring_buf, ring_head, ring_tail, coincidenceWindow):
    
    ring_size = ring_buf.size
    n_groups = g_bitmap.size

    for tag in tags:
        if tag["type"] != 0:  
            # Skip any tags that are not normal time tag.
            continue

        if tag['channel'] < -18 or tag['channel'] > 18:
            # Ignore non-physical channels
            continue

        ring_buf[ring_head] = tag

        # 1. Check if we have complete window and then evaluate coincidences within
        if ring_buf[ring_head]['time'] - ring_buf[ring_tail]['time'] > coincidenceWindow:
            # 1.1 Make channel bitmap within the complete window
            window_bitmap = 0
            # Number of elements in the ring
            ring_used = (ring_head-ring_tail) & (ring_size-1)
            for i in range(ring_used):
                buf_pos = (ring_tail + i) % ring_size
                chan = ring_buf[buf_pos]['channel']
                window_bitmap = window_bitmap | (1<< (chan + chan_offset))

            # 1.2 Compare window bitmap with the group bitmap and increment counter.
            for group_i in range(n_groups):
                if (window_bitmap & g_mask[group_i]) == g_bitmap[group_i]:
                    counter[group_i] += 1
            
            # 1.3. Shift the window
            while ring_buf[ring_head]['time'] - ring_buf[ring_tail]['time'] > coincidenceWindow:
                ring_tail = (ring_tail + 1) % ring_size

        # 2. Increment ring buffer head index.
        ring_head = (ring_head + 1) % ring_size

    return ring_head, ring_tail


###########################################################################################################
# EXAMPLE CODE BEGINS HERE
###########################################################################################################

if __name__ == '__main__':

    coin_groups = [
        ([1, 2, 3], []),                # Coincidence "1 and 2 and 3"
        Group([1, 2], [3]),             # Coincidence "1 and 2 and not 3"
        Group(incl=[1, 2], excl=[6]),   # Coincidence "1 and 2 and not 6"
    ]

    tt = TimeTagger.createTimeTaggerVirtual()
    tt.setReplaySpeed(1)
    tt.setTestSignal(1, True)
    tt.setTestSignal(2, True)
    tt.setTestSignal(3, True)
    tt.setTestSignal(6, False)

    ecc = ExclusiveCoincidenceCounter(tt, coin_groups, coincidenceWindow=1000)

    print('Coincidence groups described as bitmap and bit mask.')
    for i, bitmap in enumerate(ecc._coin_group_bitmap):
        print(f'Group:  {ecc._coincidenceGroups[i]}')
        print(f'\t -> bitmap:  {bitmap:064_b}')
        print(f'\t -> mask:    {ecc._coin_group_mask[i]:064_b}')

    ecc.startFor(1e12)
    ecc.waitUntilFinished()

    capture_duration = ecc.getCaptureDuration() * 1e-12 # in seconds
    coincidences_count = ecc.getData()
    coincidences_rate = coincidences_count / capture_duration

    print(f'Results for capture duration: {capture_duration:0.2f} s')
    for i, group in enumerate(coin_groups):
        print(
            f'\tEvents in {group[0]} but without events in {group[1]}:'
            f'\t {coincidences_count[i]}, Rate: {coincidences_rate[i]:0.1f} cps'
        )