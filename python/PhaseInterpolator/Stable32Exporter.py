import TimeTagger as TT
import numpy as np
from datetime import datetime

class Stable32Exporter(TT.CustomMeasurement):
    def __init__(self, tagger, clock_channel, filename, tau = -1, divider = 1):
        TT.CustomMeasurement.__init__(self, tagger)
        self.clock_channel = clock_channel

        self.tau = tau
        self.dt_first = None
        self.elements = 0

        self.divider = divider
        self.dt_array = np.zeros(0, dtype=np.int64)

        self.register_channel(channel=clock_channel)

        np.seterr(divide='ignore', invalid='ignore')

        self.file = open(filename, 'w')
        self.filename = filename

        self.incoming_tags = None
        self.dt = None
        self.finalize_init()

    def __del__(self):
        self.stop()

    def process(self, incoming_tags, begin_time, end_time):
        assert np.all(incoming_tags['type'] == 0), "overflow handling is not implemented, please use EventDivider"

        clock_tags = incoming_tags['channel'] == self.clock_channel
        dt = incoming_tags['time'][clock_tags]

        if dt.size <= 0:
            return

        # Sum up some timestamps as low pass filter and downsample them.
        # Avoid "mean" instead of "sum" to stay compatible with integer math.
        # Yeah, rect filter is a terrible LPF, however this matches the definition of MDEV/TDEV.
        if self.divider > 1:
            self.dt_array = np.append(self.dt_array, dt)
            N = (self.dt_array.size // self.divider) * self.divider
            if N == 0:
                return
            dt = np.sum(self.dt_array[:N].reshape((-1, self.divider)), axis=1, dtype=np.int64)
            self.dt_array = self.dt_array[N:]
            if dt.size <= 0:
                return

        # Initialize everything
        if self.dt_first is None:
            self.dt_first = dt[0]
            self.elements = 0

            if self.tau < 0:
                self.tau = round((dt[1] - dt[0]) / self.divider**2)

            # Write the header after the first tag
            self.file.write(
                "File: {filename:s}\nDate: {now:s}\nType: Phase\nTau: {tau:.6e}\n# Header End\n".format(
                    filename=self.filename, tau=self.tau*self.divider*1e-12, now=datetime.now().strftime("%m/%d/%y %H:%M:%S")
                )
            )

        dt -= self.dt_first + np.arange(self.elements, self.elements + dt.size) * (self.tau * self.divider**2)
        self.elements += dt.size

        np.savetxt(self.file, dt * (1e-12 / self.divider))

if __name__ == '__main__':
    import sys
    tt = TT.createTimeTaggerVirtual()
    exporter = Stable32Exporter(tt, 5, 'test.phd', 10_000_000, 1000)
    tt.replay(sys.argv[1])
    tt.waitForCompletion()