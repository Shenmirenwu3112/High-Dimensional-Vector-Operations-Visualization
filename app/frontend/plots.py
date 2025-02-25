### plots.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from operations import vector_operations as vo
from backend.data_manager import init_data, Basis, init_angle

plt.rcParams['animation.embed_limit'] = 100.0

def initialize_plot(fig, vec_dimension, basis_data, scale, method, Basis, target_Basis):
    gs = GridSpec(2, 1, figure=fig)

    # Top subplot for Basis Vectors
    ax_top = fig.add_subplot(gs[0])
    ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_top.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_top.set_title("Basis Vectors and Interpolations", pad=20)
    ax_top.set_xlabel("Dimension")
    ax_top.set_ylabel("Module Length")

    # Interpolated lines for smoother animation
    x_interp_line = np.linspace(vec_dimension.min(), vec_dimension.max(), scale)
    Basis_interp_line = [
        vo.interpolate(vec_dimension, y, scale=scale, method=method, ret='Y') for y in Basis
    ]
    target_interp_line = [
        vo.interpolate(vec_dimension, y, scale=scale, method=method, ret='Y') for y in target_Basis
    ]

    # Bottom subplot for Angles
    ax_angles = fig.add_subplot(gs[1])
    ax_angles.set_aspect('equal')
    ax_angles.axis('off')
    ax_angles.set_title("Angles Between Basis Vectors and Complementary Subspaces", pad=20)

    return ax_top, ax_angles, Basis_interp_line, target_interp_line, x_interp_line

def update_plot(ax, vec_dimension, x_interp_line, Basis_interp_line, ax_angles, angle_basis_complementary, target_angles_basis_complementary, fig):
    scatter = []
    line_ori = []
    line_interp = []

    for i, y in enumerate(Basis):
        color = plt.cm.tab20(i % 20)
        scatter_o = ax.scatter(
            vec_dimension, y, color=color, alpha=0.6, picker=True,
            # label=f'Data_Points_{i + 1}'
        )
        line_o, = ax.plot(
                vec_dimension, y, color=color, alpha=0.6, picker=True,
                label=f'Basis_{i + 1}'
            )
        line_i, = ax.plot(
                x_interp_line, Basis_interp_line[i], color=color, alpha=0.6, picker=True,
                # label=f'Basis_interpolation_{i + 1}'
            )

        scatter.append(scatter_o)
        line_ori.append(line_o)
        line_interp.append(line_i)



    angle_plots = []
    max_y_values = []
    offsets = []

    for i in range(Basis.shape[0]):
        offset = get_arc_offset(i, Basis.shape[0])
        offsets.append(offset)
        arc, line1, line2, angle_text, max_y = init_angle(
            ax_angles, i, angle_basis_complementary[i], offset=offset)

        angle_plots.append((
            arc, line1, line2, angle_text, 
            angle_basis_complementary[i], 
            target_angles_basis_complementary[i]
        ))

        max_y_values.append(max_y)

    # Adjust the y-limits of the subplot to include all arcs
    y_min = min(offset[1] for offset in offsets) - 0.5
    y_max = max(max_y_values) + 0.5
    ax_angles.set_ylim(y_min, y_max)

    # Adjust the x-limits similarly if needed
    x_min = min(offset[0] for offset in offsets) - 0.5
    x_max = max(offset[0] for offset in offsets) + 1.0
    ax_angles.set_xlim(x_min, x_max)

    return scatter, line_ori, line_interp, ax_angles, angle_plots


def get_arc_offset(i, max_arcs):
        '''
        Arrange arcs with a specific number of columns, displaying 1-5 at the top and 6-10 at the bottom.
        '''
        n_cols = max_arcs
        padding = 1 

        # Calculate row and column indices
        row = i // n_cols
        col = i % n_cols

        # Invert the row index to start from the top
        total_rows = (max_arcs + n_cols - 1) // n_cols  # Ceiling division
        inverted_row = total_rows - 1 - row

        x_offset = col * padding
        y_offset = inverted_row * padding
        return (x_offset, y_offset)

def create_canvas(fig):
    return FigureCanvas(fig)

def update_angle(frame, total_frames, ax, i, arc, line1, line2, angle_text, initial_angle, target_angle, radius=0.2, offset=(0, 0)):
    """
    Updates the angle plot (arc) for the current frame.

    Parameters:
        frame (int): Current frame number.
        total_frames (int): Total number of frames in the animation.
        ax (matplotlib.axes.Axes): The axis containing the arc.
        i (int): Index of the basis vector.
        arc (matplotlib.patches.Arc): The arc representing the angle.
        line1, line2 (matplotlib.lines.Line2D): Lines representing the angle.
        angle_text (matplotlib.text.Text): Text annotation for the angle.
        initial_angle (float): Initial angle in degrees.
        target_angle (float): Target angle in degrees.
        radius (float): Radius of the arc.
        offset (tuple): (x, y) offset to separate arcs.

    Returns:
        List of updated plot elements.
    """
    alpha = frame / (total_frames - 1) if total_frames > 1 else 1.0
    new_angle = (1 - alpha) * initial_angle + alpha * target_angle

    # Update the arc
    arc.theta2 = new_angle

    # Update the lines representing the angle
    x1 = offset[0] + radius * np.cos(np.radians(new_angle))
    y1 = offset[1] + radius * np.sin(np.radians(new_angle))
    line2.set_data([offset[0], x1], [offset[1], y1])

    # Update the angle text
    angle_text.set_text(f'{i + 1}: {new_angle:.2f}°')

    return [arc, line1, line2, angle_text]

def radar_factory(num_vars):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta

def create_radar_graph(fig, data_start, data_end, labels, total_frames=100):
    num_vars = len(labels)
    theta = radar_factory(num_vars)

    data_start = np.concatenate((data_start, data_start[:, :1]), axis=1)
    data_end = np.concatenate((data_end, data_end[:, :1]), axis=1)
    theta = np.append(theta, theta[0])

    ax = fig.add_subplot(projection='radar')
    ax.set_varlabels(labels)
    max_val = max(np.max(data_start), np.max(data_end))
    ax.set_ylim(0, max_val + 10)

    lines = [ax.plot(theta, row, linewidth=2, label=f'Basis {i + 1}')[0] for i, row in enumerate(data_start)]
    fills = [ax.fill(theta, row, alpha=0.25)[0] for row in data_start]

    return {
        'lines': lines,
        'fills': fills,
        'ax': ax,
        'theta': theta,
        'data_start': data_start,
        'data_end': data_end
    }