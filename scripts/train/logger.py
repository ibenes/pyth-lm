# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO         # Python 3.x


class Logger(object):
    def __init__(self, log_dir, update_freq):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.update_freq = update_freq
        self.step = 0

    def next_step(self):
        self.step += 1

    def logging_step(self):
        return (self.step+1) % self.update_freq == 0

    def scalar_summary(self, tag, value, enforce=False):
        if not enforce and not self.logging_step():
            return

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, self.step)

    def image_summary(self, tag, images, enforce=False):
        if not enforce and not self.logging_step():
            return

        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, self.step)

    def histo_summary(self, tag, values, bins=1000, enforce=False):
        if not enforce and not self.logging_step():
            return

        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()
