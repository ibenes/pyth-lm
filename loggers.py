import sys
import time
import math


class ProgressLogger():
    def __init__(self, epoch, report_period, lr, nb_updates, output_file=sys.stdout):
        self._start_time = time.time()
        self._nb_logs = 0
        self._running_loss = 0.0
        self._epoch = epoch
        self._report_period = report_period
        self._of = output_file
        self._lr = lr
        self._construction_time = time.time()
        self._nb_updates = nb_updates

    def log(self, loss):
        self._running_loss += loss
        self._nb_logs += 1 

        if self._nb_logs % self._report_period == 0:
            self._flush()
            self._reset()

    def time_since_creation(self):
        return time.time() - self._construction_time

    def nb_updates(self):
        return self._nb_updates

    def _flush(self):
        ms_per_log = (time.time() - self._start_time) * 1000 / self._report_period
        cur_loss = (self._running_loss / self._report_period)[0]
        fmt_string = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:.3e} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}\n'
        line = fmt_string.format(
            self._epoch, self._nb_logs, self._nb_updates, self._lr,
            ms_per_log, cur_loss, math.exp(cur_loss)
        )
        self._of.write(line)

    def _reset(self):
        self._running_loss = 0.0
        self._start_time = time.time()


class InfinityLogger():
    def __init__(self, epoch, report_period, lr, output_file=sys.stdout):
        self._start_time = time.time()
        self._nb_logs = 0
        self._running_loss = 0.0
        self._epoch = epoch
        self._report_period = report_period
        self._of = output_file
        self._lr = lr
        self._construction_time = time.time()

    def log(self, loss):
        self._running_loss += loss
        self._nb_logs += 1 

        if self._nb_logs % self._report_period == 0:
            self._flush()
            self._reset()

    def time_since_creation(self):
        return time.time() - self._construction_time

    def nb_updates(self):
        return self._nb_logs

    def _flush(self):
        ms_per_log = (time.time() - self._start_time) * 1000 / self._report_period
        cur_loss = (self._running_loss / self._report_period)[0]
        fmt_string = '| epoch {:3d} | {:5d} batches done | lr {:.3e} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}\n'
        line = fmt_string.format(
            self._epoch, self._nb_logs, self._lr,
            ms_per_log, cur_loss, math.exp(cur_loss)
        )
        self._of.write(line)

    def _reset(self):
        self._running_loss = 0.0
        self._start_time = time.time()
