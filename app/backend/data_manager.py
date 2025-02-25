### backend_data_processing.py
import numpy as np
from operations import vector_operations as vo
from operations import lattice_operations as lo

# Example Basis (10x10 Basis Matrix)
Basis = np.random.randint(0, 1000, size=(10,10))

def init_data(Basis = Basis):
    """
    Initializes data by performing LLL reduction and calculating angles.

    Parameters:
        Basis (np.ndarray): The original basis matrix.

    Returns:
        data (dict): Dictionary containing original and reduced basis, angles, and other parameters.
    """
    target_Basis = lo.LLL_reduction(Basis.copy())
    vec_dimension = np.arange(1, Basis.shape[1] + 1)
    scale = 100  # Adjusted for smoother interpolation
    method = 'cubicspline'
    total_frames = 100  # Number of animation frames

    angles_basis_complementary = vo.calculate_angles_with_complementary(Basis)
    target_angles_basis_complementary = vo.calculate_angles_with_complementary(target_Basis)

    # Ensure angles are finite
    angles_basis_complementary = np.nan_to_num(
        angles_basis_complementary, nan=0.0, posinf=0.0, neginf=0.0
    )
    target_angles_basis_complementary = np.nan_to_num(
        target_angles_basis_complementary, nan=0.0, posinf=0.0, neginf=0.0
    )

    return {
        'target_Basis': np.abs(target_Basis),
        'vec_dimension': vec_dimension,
        'scale': scale,
        'method': method,
        'angles_basis_complementary': angles_basis_complementary,
        'target_angles_basis_complementary': target_angles_basis_complementary,
        'total_frames': total_frames
    }

def init_angle(ax, i, angle_deg, radius=0.4, offset=(0, 0)):
    """
    Initializes the angle plot (arc) for the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot the arc.
        i (int): Index of the basis vector.
        angle_deg (float): Initial angle in degrees.
        radius (float): Radius of the arc.
        offset (tuple): (x, y) offset to separate arcs.

    Returns:
        arc, line1, line2, angle_text, max_y: Plot elements and maximum y-value for the angle visualization.
    """
    from matplotlib.patches import Arc

    # Create an arc representing the angle
    arc = Arc(
        (offset[0], offset[1]),
        2 * radius,
        2 * radius,
        theta1=0,
        theta2=angle_deg,
        color='b',
        linewidth=2,
        picker=True,
        alpha=0.6
    )

    # Create lines to represent the angle
    line1, = ax.plot(
        [offset[0], offset[0] + radius],
        [offset[1], offset[1]],
        color='r',
        lw=2,
        alpha=0.6
    )
    x1 = offset[0] + radius * np.cos(np.radians(angle_deg))
    y1 = offset[1] + radius * np.sin(np.radians(angle_deg))
    line2, = ax.plot(
        [offset[0], x1],
        [offset[1], y1],
        color='g',
        lw=2,
        alpha=0.6
    )

    # Text annotation for the angle
    angle_text = ax.text(
        offset[0],
        offset[1] - 0.2,
        f'{i + 1}: {angle_deg:.2f}°',
        ha='center',
        va='center',
        fontsize=8,
        alpha=0.8
    )

    # Add the arc to the axis
    ax.add_patch(arc)
    ax.set_aspect("equal")
    ax.axis('off')

    # Compute the maximum y-value used
    max_y = max(offset[1], y1)

    return arc, line1, line2, angle_text, max_y

def smoothstep(t):
    """
    Smoothstep function for easing animation transitions.
    """
    return t * t * (3 - 2 * t)