#!/usr/bin/env python

from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

class TFMPLFigure(object):
    def __init__(self, name, writer):
        """Create an interface to rasterize matplotlib figures to tensorflow Summaries."""

        # Create placeholder for the figure image
        self.placeholder = tf.placeholder(tf.uint8, (None, None, None, None))

        # Create image summary of the figure image
        self.summary = tf.summary.image(name, self.placeholder, max_outputs=1)
        self.writer = writer
        self.counter = 0

    def rasterize(self, fig):
        """Given a figure from self.get_figure(), rasterize it and add it to the summary."""
        image = TFMPLFigure.fig2rgb_array(fig)
        summary = self.summary.eval(feed_dict={self.placeholder: image})
        self.writer.add_summary(summary, global_step=self.counter)
        self.counter += 1

    @staticmethod
    def fig2rgb_array(fig, expand=True):
        """Convert an mpl to a numpy array"""
        buf_ = BytesIO()
        fig.savefig(buf_, format='png')
        buf_.seek(0)
        image = Image.open(buf_)

        # Increate rank because tf.summmary.image expects a 4d tensor.
        npa = np.asarray(image)[None]
        buf_.close()
        return npa

if __name__ == '__main__':
    summary_writer = tf.summary.FileWriter('./tmp')
    tffig0 = TFMPLFigure('large', summary_writer)
    tffig1 = TFMPLFigure('small', summary_writer)    

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(10):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), frameon=False)
            image0 = 1 * np.random.random((100, 100, 3))
            image1 = 10 * np.random.random((50, 50 * (i + 1), 3))
            axes[0].imshow(image0)
            axes[1].imshow(image1)

            tffig0.rasterize(fig)

            fig, axes = plt.subplots(1, 1, figsize=(6, 6), frameon=False)
            image3 = np.random.random((10, 5, 3))
            axes.imshow(image3)
            tffig1.rasterize(fig)
