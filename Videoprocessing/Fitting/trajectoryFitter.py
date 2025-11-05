"""
Created on Jan 30 2025 15:45

@author: ISAC - pettirsch
"""

import numpy as np
import scipy.interpolate as si
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BPoly
import math

from scipy.optimize import leastsq


class TrajectoryFitter:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit_finished_tracks(self, active_tracks, finished_track_ids):

        for track_id in finished_track_ids:
            # Get Track
            track = active_tracks[track_id]

            # Preprocess world positions
            worldPositions = track.get_world_positions()

            # Smooth trajectory
            # worldPositions_smooth = self.smooth_trajectory(worldPositions)
            worldPositions_smooth = self.smooth_trajectory_moving_average(worldPositions)

            smoothed_px = worldPositions_smooth[:, 0]
            smoothed_py = worldPositions_smooth[:, 1]
            smoothed_pz = worldPositions_smooth[:, 2]
            track.set_world_positions_fitted(worldPositions_smooth)

            smoothed_vx = np.gradient(smoothed_px)*30
            smoothed_vy = np.gradient(smoothed_py)*30
            smoothed_vz = np.gradient(smoothed_pz)*30
            velocities_smoothed = np.vstack((smoothed_vx, smoothed_vy, smoothed_vz)).T
            track.set_velocities_fitted(velocities_smoothed)

            smoothed_ax = np.gradient(smoothed_vx)
            smoothed_ay = np.gradient(smoothed_vy)
            smoothed_az = np.gradient(smoothed_vz)
            accelerations_smoothed = np.vstack((smoothed_ax, smoothed_ay, smoothed_az)).T
            track.set_accelerations_fitted(accelerations_smoothed)

            smoothed_yaw = np.arctan2(smoothed_vy, smoothed_vx)
            distance = 0
            start_idx = 0
            for i in range(len(worldPositions_smooth)):
                if i > 1:
                    distance = np.linalg.norm(worldPositions_smooth[i] - worldPositions_smooth[start_idx])
                if distance > 1:
                    yaw = worldPositions_smooth[i,:] - worldPositions_smooth[start_idx,:]
                    yaw = np.arctan2(yaw[1], yaw[0])
                    for j in range(start_idx, i):
                        smoothed_yaw[j] = yaw
                    start_idx = i
                    distance = 0

            def moving_average(data, window_size):
                """Applies a moving average filter with proper edge handling."""
                if window_size < 2:
                    return data  # No smoothing needed for window_size < 2

                # Ensure the window size is odd for symmetric smoothing
                if window_size % 2 == 0:
                    window_size += 1

                pad_size = window_size // 2
                padded_data = np.pad(data, (pad_size, pad_size), mode='edge')  # Replicate edges
                return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

            smoothed_yaw2 = active_tracks[track_id].history['yaws'].copy()

            def detect_jumps(lst):
                return [i for i in range(len(lst) - 1) if abs(lst[i + 1] - lst[i]) > 3]

            # Example usage:
            jumps = detect_jumps(smoothed_yaw2)
            start_idx = 0
            if len(jumps) > 0:
                for q, jump in enumerate(jumps):
                    window_size = min(jump - start_idx, 60)
                    smoothed_yaw2[start_idx:jump+1] = moving_average(smoothed_yaw2[start_idx:jump+1], window_size)
                    start_idx = jump
                    if q == len(jumps) - 1:
                        window_size = min(len(smoothed_yaw2) - start_idx, 60)
                        smoothed_yaw2[start_idx:] = moving_average(smoothed_yaw2[start_idx:], window_size)
            else:
                window_size = min(len(smoothed_yaw2) - start_idx, 60)
                smoothed_yaw2[start_idx:] = moving_average(smoothed_yaw2[start_idx:], window_size)

            track.set_yaws_fitted(smoothed_yaw)
            smoothed_yaw_rate = np.gradient(smoothed_yaw)
            track.set_yaw_rates_fitted(smoothed_yaw_rate)

            # Draw X Y of world positions and smoothed positions in different colors as lines
            # if track_id > 16:
            #     print(track_id)
            #     worldPositions = np.array(worldPositions)
            #     from matplotlib import pyplot as plt
            #     import matplotlib
            #     matplotlib.use('TkAgg')
            #     plt.plot(worldPositions[:, 0], worldPositions[:, 1], 'r')
            #     plt.plot(worldPositions_smooth[:, 0], worldPositions_smooth[:, 1], 'b')
            #     # plt.plot(worldPositions_smooth_2[:, 0], worldPositions_smooth_2[:, 1], 'g')
            #     # plt.plot(smoothed_px, smoothed_py, 'g')
            #     plt.show()
            #     # Wait
            #     input("Press Enter to continue...")
            #     plt.close()
            #
            #     # Plot speeds
            #     # raw_vx = np.array(active_tracks[track_id].history['vel_x'])
            #     # raw_vy = np.array(active_tracks[track_id].history['vel_y'])
            #     # raw_vz = np.array(active_tracks[track_id].history['vel_z'])
            #     # plt.plot(raw_vx, 'r')
            #     # plt.plot(raw_vy, 'g')
            #     # plt.plot(raw_vz, 'b')
            #     #
            #     # # plot smmoth in purple, yellow, cyan
            #     # plt.plot(smoothed_vx, 'm')
            #     # plt.plot(smoothed_vy, 'y')
            #     # plt.plot(smoothed_vz, 'c')
            #     #
            #     # plt.show()
            #     # # Wait
            #     # input("Press Enter to continue...")
            #     # plt.close()
            #     #
            #     # plt.plot(smoothed_ax, 'm')
            #     # plt.plot(smoothed_ay, 'y')
            #     # plt.plot(smoothed_az, 'c')
            #     #
            #     # plt.show()
            #     # # Wait
            #     # input("Press Enter to continue...")
            #     # plt.close()
            #
            #
            #
            #     # plot yaws
            #     raw_yaw = np.array(active_tracks[track_id].history['yaws'])
            #     plt.plot(raw_yaw, 'r')
            #     plt.plot(smoothed_yaw, 'b')
            #     plt.plot(smoothed_yaw2, 'g')
            #     plt.show()
            #     # Wait
            #     input("Press Enter to continue...")
            #     plt.close()

        return active_tracks, finished_track_ids

    def adaptive_segmentation(self, positions, curvature_threshold=10):
        """
            Segments trajectory adaptively based on curvature.
            """

        positions = np.array(positions)
        segments = []
        current_segment = []

        for i in range(len(positions)):
            current_segment.append(positions[i])

            if len(current_segment) >= 4:
                curvatures = self.compute_curvature(np.array(current_segment)[:, 0], np.array(current_segment)[:, 1])

                if np.any(curvatures > curvature_threshold):
                    segments.append(current_segment[:-1])  # Keep last point for next segment
                    current_segment = [current_segment[-1]]

        if len(current_segment) >= 4:
            segments.append(current_segment)

        return segments

    def distance_based_segmentation(self, positions, distance_threshold=3, min_segment_length=6):
        """
        Segments trajectory when the cumulative distance exceeds a threshold.
        """
        positions = np.array(positions)
        segments = []
        current_segment = [positions[0]]
        total_distance = 0

        for i in range(1, len(positions)):
            step_distance = np.linalg.norm(positions[i] - positions[i - 1])
            total_distance += step_distance

            if total_distance > distance_threshold and len(current_segment) >= min_segment_length:
                segments.append(current_segment)
                current_segment = [positions[i]]
                total_distance = 0  # Reset distance counter
            else:
                current_segment.append(positions[i])

        if len(current_segment) >= min_segment_length:
            segments.append(current_segment)

        return segments

    def remove_trajectory(self, traj_id, record_id):
        """
        Removes a trajectory from the database.
        """
        print("test")

    def optimal_sigma(self, positions, min_sigma = 0.5, max_sigma = 5 ) :
        """
        Estimates a good sigma value for Gaussian filtering based on trajectory variation.
        """
        """
           Estimates a good sigma value for Gaussian filtering based on trajectory variation.
           Ensures sigma remains in a reasonable range.
           """
        positions = np.array(positions)
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # Compute step distances

        # Use interquartile range (IQR) to estimate typical step size
        q1, q3 = np.percentile(diffs, [25, 75])
        iqr = q3 - q1

        # Estimate sigma based on IQR instead of mean/std (robust to noise)
        sigma = 15 * iqr

        # Clamp sigma within reasonable range
        return max(min_sigma, min(sigma, max_sigma))

    def smooth_trajectory(self, positions, sigma=None):
        """
        Smooths the trajectory using a Gaussian filter with an adaptive sigma.
        """
        positions = np.array(positions)
        if sigma is None:
            sigma = self.optimal_sigma(positions)  # Auto-tune sigma

        smoothed_x = gaussian_filter1d(positions[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(positions[:, 1], sigma=sigma)

        if positions.shape[1] > 2:
            smoothed_z = gaussian_filter1d(positions[:, 2], sigma=sigma)
            return np.vstack((smoothed_x, smoothed_y, smoothed_z)).T
        else:
            return np.vstack((smoothed_x, smoothed_y)).T

    # def smooth_trajectory(self, positions, sigma=1.5):
    #     """
    #     Smooths the trajectory using a Gaussian filter.
    #     """
    #     sigma = 3
    #
    #     positions = np.array(positions)
    #     smoothed_x = gaussian_filter1d(positions[:, 0], sigma=sigma)
    #     smoothed_y = gaussian_filter1d(positions[:, 1], sigma=sigma)
    #     smoothed_z = gaussian_filter1d(positions[:, 2], sigma=sigma) if positions.shape[1] > 2 else np.zeros_like(
    #         smoothed_x)
    #
    #     return np.vstack((smoothed_x, smoothed_y, smoothed_z)).T

    def smooth_trajectory_moving_average(self,positions, window_size=10):
        """
        Smooths the trajectory using a moving average filter with proper edge handling.

        :param positions: List or array of (x, y, z) or (x, y) coordinates.
        :param window_size: Size of the moving average window (should be odd).
        :return: Smoothed trajectory as a NumPy array.
        """

        def moving_average(data, window_size):
            """Applies a moving average filter with proper edge handling."""
            if window_size < 2:
                return data  # No smoothing needed for window_size < 2

            # Ensure the window size is odd for symmetric smoothing
            if window_size % 2 == 0:
                window_size += 1

            pad_size = window_size // 2
            padded_data = np.pad(data, (pad_size, pad_size), mode='edge')  # Replicate edges
            return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

        positions = np.array(positions)

        smoothed_x = moving_average(positions[:, 0], window_size)
        smoothed_y = moving_average(positions[:, 1], window_size)

        if positions.shape[1] > 2:
            smoothed_z = moving_average(positions[:, 2], window_size)
            return np.vstack((smoothed_x, smoothed_y, smoothed_z)).T
        else:
            return np.vstack((smoothed_x, smoothed_y)).T

    def angle_based_segmentation(self, positions, angle_threshold=20, min_segment_length=5):
        """
        Segments trajectory based on changes in direction (angle differences).
        """
        positions = np.array(positions)
        segments = []
        current_segment = [positions[0], positions[1]]

        for i in range(2, len(positions)):
            v1 = positions[i - 1] - positions[i - 2]
            v2 = positions[i] - positions[i - 1]
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norm_product == 0:
                angle = 0
            else:
                angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))

            if angle > angle_threshold and len(current_segment) >= min_segment_length:
                segments.append(current_segment)
                current_segment = [positions[i - 1], positions[i]]
            else:
                current_segment.append(positions[i])

        if len(current_segment) >= min_segment_length:
            segments.append(current_segment)

        return segments

    def compute_curvature(self, x, y):
        """
        Computes the curvature of a trajectory given x and y coordinates.
        """
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
        return curvature

    # def fit_constrained_b_spline(self, positions, x_start=None, vx_start=None, ax_start=None, y_start=None,
    #                              vy_start=None, ay_start=None,
    #                              z_start=None, vz_start=None, az_start=None):
    #     """
    #     Fits a B-spline while enforcing continuity in position, first derivative (velocity),
    #     and second derivative (acceleration) between segments.
    #
    #     - Uses previous segment's velocity & acceleration at the start for smooth merging.
    #     """
    #
    #     positions = np.array(positions)
    #     num_points = len(positions)
    #     assert num_points > 4
    #
    #     degree = 3  # Cubic B-spline
    #
    #     # Parameterize the curve
    #     u_original = np.linspace(0, 1, num_points)
    #
    #     # Compute derivatives at start
    #     d_start_x = vx_start if vx_start is not None else (positions[1, 0] - positions[0, 0]) / (
    #                 u_original[1] - u_original[0])
    #     dd_start_x = ax_start if ax_start is not None else 0
    #     d_start_y = vy_start if vy_start is not None else (positions[1, 1] - positions[0, 1]) / (
    #                 u_original[1] - u_original[0])
    #     dd_start_y = ay_start if ay_start is not None else 0
    #     d_start_z = vz_start if vz_start is not None else (positions[1, 2] - positions[0, 2]) / (
    #                 u_original[1] - u_original[0])
    #     dd_start_z = az_start if az_start is not None else 0
    #
    #     # Fit B-spline with start boundary constraints over x, y, z
    #     tck_x, _ = si.splprep([positions[:, 0]], k=degree, s=0.1, bc_type=([(x_start, d_start_x, dd_start_x)], None))
    #     tck_y, _ = si.splprep([positions[:, 1]], k=degree, s=0.1, bc_type=([(y_start, d_start_y, dd_start_y)], None))
    #     tck_z, _ = si.splprep([positions[:, 2]], k=degree, s=0.1, bc_type=([(z_start, d_start_z, dd_start_z)], None))
    #
    #     # Get smoothed positions
    #     x_smooth = si.splev(u_original, tck_x)
    #     y_smooth = si.splev(u_original, tck_y)
    #     z_smooth = si.splev(u_original, tck_z)
    #
    #     # Get smoothed velocities
    #     vx_smooth = si.splev(u_original, tck_x, der=1)
    #     vy_smooth = si.splev(u_original, tck_y, der=1)
    #     vz_smooth = si.splev(u_original, tck_z, der=1)
    #
    #     # Get smoothed accelerations
    #     ax_smooth = si.splev(u_original, tck_x, der=2)
    #     ay_smooth = si.splev(u_original, tck_y, der=2)
    #     az_smooth = si.splev(u_original, tck_z, der=2)
    #
    #     yaws_smooth = np.arctan2(vy_smooth, vx_smooth)
    #     yaw_rates_smooth = np.gradient(yaws_smooth)

    def fit_bpoly_with_constraints(self, segment, px_start=None, vx_start=None, ax_start=None, py_start=None,
                                   vy_start=None, ay_start=None, pz_start=None, vz_start=None, az_start=None,
                                   degree=5):

        # Extract position data from segment
        segment = np.asarray(segment)
        pos_x, pos_y, pos_z = segment[:, 0], segment[:, 1], segment[:, 2]

        # Define breakpoints (time parameter t for interpolation)
        xi = np.linspace(0, 1, len(segment))  # Normalize time to [0, 1]

        # Use provided constraints or default to segment start
        px_start = px_start if px_start is not None else pos_x[0]
        py_start = py_start if py_start is not None else pos_y[0]
        pz_start = pz_start if pz_start is not None else pos_z[0]

        # Compute finite difference derivatives if not provided
        def estimate_derivatives(pos):
            v = np.gradient(pos, xi)  # First derivative (velocity)
            a = np.gradient(v, xi)  # Second derivative (acceleration)
            return v, a

        vx, ax = estimate_derivatives(pos_x)
        vy, ay = estimate_derivatives(pos_y)
        vz, az = estimate_derivatives(pos_z)

        vx_start = vx_start if vx_start is not None else vx[0]
        vy_start = vy_start if vy_start is not None else vy[0]
        vz_start = vz_start if vz_start is not None else vz[0]

        ax_start = ax_start if ax_start is not None else ax[0]
        ay_start = ay_start if ay_start is not None else ay[0]
        az_start = az_start if az_start is not None else az[0]

        px_end, py_end, pz_end = pos_x[-1], pos_y[-1], pos_z[-1]
        vx_end, vy_end, vz_end = vx[-1], vy[-1], vz[-1]
        ax_end, ay_end, az_end = ax[-1], ay[-1], az[-1]

        # Collect constraints for Bernstein polynomial
        yi_x = [[px_start, vx_start, ax_start]]  # First point with full constraints
        yi_y = [[py_start, vy_start, ay_start]]
        yi_z = [[pz_start, vz_start, az_start]]

        # Interior points: Only position constraints
        for i in range(1, len(segment) - 1):
            yi_x.append([pos_x[i], vx[i], ax[i]])
            yi_y.append([pos_y[i], vy[i], ay[i]])
            yi_z.append([pos_z[i], vz[i], az[i]])

        # Last point: Full constraints again (to enforce smooth endpoint)
        yi_x.append([px_end, vx_end, ax_end])
        yi_y.append([py_end, vy_end, ay_end])
        yi_z.append([pz_end, vz_end, az_end])

        # Fit the BPoly with constraints
        BPoly_x = BPoly.from_derivatives(xi, yi_x, orders=degree)
        BPoly_y = BPoly.from_derivatives(xi, yi_y, orders=degree)
        BPoly_z = BPoly.from_derivatives(xi, yi_z, orders=degree)

        # Get fitted values
        x_fitted = BPoly_x(xi)
        y_fitted = BPoly_y(xi)
        z_fitted = BPoly_z(xi)

        # Get velocities
        vx_fitted = BPoly_x(xi, nu=1)
        vy_fitted = BPoly_y(xi, nu=1)
        vz_fitted = BPoly_z(xi, nu=1)

        # Get accelerations
        ax_fitted = BPoly_x(xi, nu=2)
        ay_fitted = BPoly_y(xi, nu=2)
        az_fitted = BPoly_z(xi, nu=2)

        # Get yaw
        yaw_fitted = np.arctan2(vy_fitted, vx_fitted)
        yaw_rate_fitted = np.gradient(yaw_fitted)

        return x_fitted, y_fitted, z_fitted, vx_fitted, vy_fitted, vz_fitted, ax_fitted, ay_fitted, az_fitted, yaw_fitted, yaw_rate_fitted

    # Function to compute a Bézier curve

    def fit_b_spline_3d(self, positions, num_samples=50):
        """
        Fits a 3D B-spline to the trajectory with an adaptive degree and handles short segments.
        """
        positions = np.array(positions)
        num_points = len(positions)
        assert num_points > 4

        degree = 3  # Cubic B-spline

        u = np.linspace(0, 1, num_points) # Resample 0-1 since num_points are fixed frequency

        # Fit B-spline
        tck_x,_ = si.splprep([positions[:, 0]], k=degree, s=0.1)
        tck_y,_ = si.splprep([positions[:, 1]], k=degree, s=0.1)
        tck_z,_ = si.splprep([positions[:, 2]], k=degree, s=0.1)

        x_smooth = si.splev(u, tck_x)[0]
        y_smooth = si.splev(u, tck_y)[0]
        z_smooth = si.splev(u, tck_z)[0]
        smoothed_positions = np.vstack((x_smooth, y_smooth, z_smooth)).T

        vx_smooth = si.splev(u, tck_x, der=1)[0]
        vy_smooth = si.splev(u, tck_y, der=1)[0]
        vz_smooth = si.splev(u, tck_z, der=1)[0]
        velocities = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T

        ax_smooth = si.splev(u, tck_x, der=2)[0]
        ay_smooth = si.splev(u, tck_y, der=2)[0]
        az_smooth = si.splev(u, tck_z, der=2)[0]
        accelerations = np.vstack((ax_smooth, ay_smooth, az_smooth)).T

        # Compute yaw (heading angle)
        yaws = np.arctan2(vy_smooth, vx_smooth)

        yaw_rates = np.gradient(yaws, u)

        return x_smooth, y_smooth, z_smooth, vx_smooth, vy_smooth, vz_smooth, ax_smooth, ay_smooth, az_smooth, yaws, yaw_rates



# def bezier_curve(t, control_points):
#     n = len(control_points) - 1
#     curve = np.zeros(2)
#     for i in range(n + 1):
#         binomial_coeff = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
#         bernstein_poly = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))
#         curve += bernstein_poly * control_points[i]
#
#
#     return curve
#
#     # Least squares error function for Bézier fitting
# def bezier_fit_error(control_points_flat, data_points, t_values, num_control_points):
#     control_points = control_points_flat.reshape(num_control_points, 2)
#     error = []
#     for i, t in enumerate(t_values):
#         fitted_point = bezier_curve(t, control_points)
#         error.append(fitted_point - data_points[i])
#     return np.ravel(error)
