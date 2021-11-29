from progress.bar import Bar


class ProgressBar(Bar):
    suffix = "%(percent).1f%%, ETA: %(eta)ds, %(frequency)fHz"

    @property
    def frequency(self):
        return 1 / self.avg
