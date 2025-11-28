import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from node import Node


class CircularGraph:

    def __init__(self, adjacency_matrix, colormap=None, labels=None):
       
        n = len(adjacency_matrix)
        print(f"\nInitializing CircularGraph with {n}x{n} matrix")
        print(f"Non-zero entries: {np.count_nonzero(adjacency_matrix)}")

        if colormap is None:
            colormap = plt.cm.viridis(np.linspace(0, 1, n))[:, :3]

        if labels is None:
            labels = [str(i + 1) for i in range(n)]

        self.original_adjacency = adjacency_matrix.copy()
        self.original_colormap = colormap.copy()
        self.labels = labels
        self.nodes = []

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.patch.set_facecolor('white')

      
        lobes_positions = [13, 30, 31, 38, 42, 46, 53, 54, 71, 85]

        print(f"\nInserting blanks at positions: {lobes_positions}")

        adj_expanded = adjacency_matrix.copy()
        colormap_expanded = colormap.copy()

        self.orig_to_expanded = {}
        offset = 0
        for i in range(n):
            self.orig_to_expanded[i] = i + offset
            if i in lobes_positions:
                offset += 1

        for i, pos in enumerate(lobes_positions):
            insert_at = pos + 1 + i  
            adj_expanded = np.insert(adj_expanded, insert_at, 0, axis=0)
            colormap_expanded = np.insert(colormap_expanded, insert_at, [1, 1, 1], axis=0)

        for i, pos in enumerate(lobes_positions):
            insert_at = pos + 1 + i
            adj_expanded = np.insert(adj_expanded, insert_at, 0, axis=1)

        self.adjacency_expanded = adj_expanded
        self.colormap_expanded = colormap_expanded

        print(f"Expanded matrix size: {adj_expanded.shape}")

        self._create_nodes()

        self.node_visibility = [True] * len(self.nodes)

        self._draw_connections()

        self._setup_axis()

        self._create_buttons()

        plt.show()

    def _create_nodes(self):
        n_expanded = len(self.adjacency_expanded)

        step = 2 * np.pi / n_expanded
        angles = np.linspace(step + np.pi / 2, step + 5 * np.pi / 2, n_expanded + 2)

        blank_positions = set()
        for orig_idx in [13, 30, 31, 38, 42, 46, 53, 54, 71, 85]:
            blank_positions.add(self.orig_to_expanded[orig_idx] + 1)

        print(f"Blank positions in expanded array: {sorted(blank_positions)}")
        print(f"Total nodes in expanded array: {n_expanded}")

        label_idx = 0
        for i in range(n_expanded):
            x = np.cos(angles[i])
            y = np.sin(angles[i])
            color = self.colormap_expanded[i]

            if i in blank_positions:
                label = ""
            else:
                if label_idx < len(self.labels):
                    label = self.labels[label_idx]
                    label_idx += 1
                else:
                    label = ""

            node = Node(x, y, color, label, self.ax)
            self.nodes.append(node)

        print(f"Created {len(self.nodes)} nodes ({label_idx} with labels, {len(blank_positions)} blanks)")

    def _draw_connections(self):
        adj = self.original_adjacency
        n_orig = len(adj)
        n_expanded = len(self.adjacency_expanded)

        rows, cols = np.where(adj != 0)
        values = adj[rows, cols]

        print(f"\n=== Drawing Connections ===")
        print(f"Original matrix: {n_orig}x{n_orig}")
        print(f"Expanded matrix: {n_expanded}x{n_expanded}")
        print(f"Non-zero entries to draw: {len(values)}")

        if len(values) == 0:
            print("WARNING: No connections to draw!")
            return

        min_line_width = 0.5
        line_width_coef = 5
        max_abs = np.max(np.abs(values))
        normalized_values = values / max_abs if max_abs > 0 else np.zeros_like(values)

        step = 2 * np.pi / n_expanded
        angles = np.linspace(step + np.pi / 2, step + 5 * np.pi / 2, n_expanded + 2)

        drawn = 0
        for i in range(len(values)):
            row_orig = rows[i]
            col_orig = cols[i]

            if row_orig == col_orig:
                continue

            row_exp = self.orig_to_expanded[row_orig]
            col_exp = self.orig_to_expanded[col_orig]

            u = np.array([np.cos(angles[row_exp]), np.sin(angles[row_exp])])
            v = np.array([np.cos(angles[col_exp]), np.sin(angles[col_exp])])

            lw = abs(line_width_coef * normalized_values[i] + min_line_width)

            if normalized_values[i] > 0:
                color = [1 - normalized_values[i], 1, 1]  
            else:
                color = [1, 1, 1 + normalized_values[i]]  

            if abs(row_exp - col_exp) == n_expanded // 2:
                line, = self.ax.plot([u[0], v[0]], [u[1], v[1]],
                                     linewidth=lw, color=color,
                                     zorder=0, alpha=0.6)
            else:
                denom = u[0] * v[1] - u[1] * v[0]
                if abs(denom) < 1e-10:
                    line, = self.ax.plot([u[0], v[0]], [u[1], v[1]],
                                         linewidth=lw, color=color,
                                         zorder=0, alpha=0.6)
                else:
                    x0 = -(u[1] - v[1]) / denom
                    y0 = (u[0] - v[0]) / denom
                    r = np.sqrt(x0 ** 2 + y0 ** 2 - 1)
                    theta1 = np.arctan2(u[1] - y0, u[0] - x0)
                    theta2 = np.arctan2(v[1] - y0, v[0] - x0)
                    if u[0] >= 0 and v[0] >= 0:
                        theta = np.concatenate([
                            np.linspace(max(theta1, theta2), np.pi, 50),
                            np.linspace(-np.pi, min(theta1, theta2), 50)
                        ])
                    else:
                        theta = np.linspace(theta1, theta2, 100)
                    line, = self.ax.plot(r * np.cos(theta) + x0,
                                         r * np.sin(theta) + y0,
                                         linewidth=lw, color=color,
                                         zorder=0, alpha=0.6)

            self.nodes[row_exp].add_connection(line)
            self.nodes[col_exp].add_connection(line)
            drawn += 1

        print(f"Successfully drew {drawn} connections")

    def _setup_axis(self):
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        extent = 0.1
        for node in self.nodes:
            if node.label:
                e = node.get_extent()
                if e > extent:
                    extent = e

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        margin = extent * 1.2
        self.ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
        self.ax.set_ylim(ylim[0] - margin * 1.75, ylim[1] + margin * 1.75)

    def _create_buttons(self):
        ax_show = plt.axes([0.4, 0.01, 0.1, 0.04])
        ax_hide = plt.axes([0.5, 0.01, 0.1, 0.04])
        self.btn_show = Button(ax_show, 'Show All')
        self.btn_hide = Button(ax_hide, 'Hide All')
        self.btn_show.on_clicked(self._on_show_all)
        self.btn_hide.on_clicked(self._on_hide_all)

        def to_exp(start, end):
            return [self.orig_to_expanded[i] for i in range(start, end + 1)]

        regions_left = [
            ('Frontal left', [0, 0.45, 0.74], 0.20, to_exp(0, 13)),
            ('Temporal left', [0.46, 0.67, 0.18], 0.15, to_exp(14, 30)),
            ('Central left', [0.85, 0.32, 0.98], 0.10, [self.orig_to_expanded[31]]),
            ('Parietal left', [0.93, 0.69, 0.12], 0.05, to_exp(32, 38)),
            ('Occipital left', [0.64, 0.08, 0.18], 0.00, to_exp(39, 42)),
        ]

        regions_right = [
            ('Frontal right', [0, 0.45, 0.74], 0.20, to_exp(72, 85)),
            ('Temporal right', [0.46, 0.67, 0.18], 0.15, to_exp(55, 71)),
            ('Central right', [0.85, 0.32, 0.98], 0.10, [self.orig_to_expanded[54]]),
            ('Parietal right', [0.93, 0.69, 0.12], 0.05, to_exp(47, 53)),
            ('Occipital right', [0.64, 0.08, 0.18], 0.00, to_exp(43, 46)),
        ]

        self.region_buttons = []

        for name, color, pos, indices in regions_left:
            ax_btn = plt.axes([0.0, pos, 0.1, 0.04])
            btn = Button(ax_btn, name, color=color, hovercolor='0.7')
            btn.label.set_color('white')
            btn.on_clicked(lambda event, idx=indices: self._on_region(idx))
            self.region_buttons.append(btn)

        for name, color, pos, indices in regions_right:
            ax_btn = plt.axes([0.9, pos, 0.1, 0.04])
            btn = Button(ax_btn, name, color=color, hovercolor='0.7')
            btn.label.set_color('white')
            btn.on_clicked(lambda event, idx=indices: self._on_region(idx))
            self.region_buttons.append(btn)

   
    def _on_show_all(self, event):
        for i, node in enumerate(self.nodes):
            node.set_visible(True)
            self.node_visibility[i] = True
        self.fig.canvas.draw_idle()

    def _on_hide_all(self, event):
        for i, node in enumerate(self.nodes):
            node.set_visible(False)
            self.node_visibility[i] = False
        self.fig.canvas.draw_idle()

    def _on_region(self, indices):
        
        for i, node in enumerate(self.nodes):
            node.set_visible(False)
            self.node_visibility[i] = False

        for i in indices:
            if i < len(self.nodes):
                self.nodes[i].set_visible(True)
                self.node_visibility[i] = True

        self.fig.canvas.draw_idle()