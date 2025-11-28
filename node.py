import numpy as np
import matplotlib.pyplot as plt


class Node:

    LABEL_OFFSET_FACTOR = 1.2  

    def __init__(self, x, y, color, label, ax):
        self.position = np.array([x, y])
        self.color = np.array(color)
        self.label = label
        self.ax = ax
        self.connections = []
        self.visible = True

        self.marker = ax.plot(
            x, y, 'o', color=color,
            markersize=8, zorder=2,
            picker=True, pickradius=5
        )[0]

        self._create_text_label()

        self.marker.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def _create_text_label(self):
        if not self.label:
            self.text_label = None
            return

        x, y = self.position
        t = np.arctan2(y, x)

        label_x = self.LABEL_OFFSET_FACTOR * x
        label_y = self.LABEL_OFFSET_FACTOR * y

        rotation_deg = np.degrees(t)

        if -90 <= rotation_deg <= 90:
            ha = 'left'
        else:
            rotation_deg = rotation_deg - 180
            ha = 'right'

        self.text_label = self.ax.text(
            label_x, label_y, self.label,
            rotation=rotation_deg,
            rotation_mode='anchor',
            horizontalalignment=ha,
            verticalalignment='center',
            fontsize=6.5,
            zorder=3
        )

    def add_connection(self, line):
        self.connections.append(line)

    def set_visible(self, visible):
        self.visible = visible
        if visible:
            self.marker.set_marker('o')
            for line in self.connections:
                line.set_color(self.color)
                line.set_zorder(1)
        else:
            self.marker.set_marker('x')
            gray = [0.9, 0.9, 0.9]
            for line in self.connections:
                line.set_color(gray)
                line.set_zorder(0)

    def get_extent(self):
        if self.text_label is None:
            return 0
        self.ax.figure.canvas.draw()
        bbox = self.text_label.get_window_extent()
        bbox_data = bbox.transformed(self.ax.transData.inverted())
        return bbox_data.width

    def _on_pick(self, event):
        if event.artist == self.marker:
            self.set_visible(not self.visible)
            self.ax.figure.canvas.draw_idle()