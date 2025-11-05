"""
Created on May 14 2025 07:44

@author: ISAC - pettirsch
"""

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import csv
from datetime import datetime

from CommonUtils.ConfigUtils.read_config import load_config
from TrajectoryProcessing.TrajectoryClusterer.trajectoryclusterer import TrajectoryClusterer
from TrajectoryProcessing.TrajectoryClusterer.trajectoryclusterer import point_to_curve_min_dist

def build_time_csv_path(config_path, output_folder):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_time_measurement_maneuver_classification_{cfg_base}.csv"
    return os.path.join(output_folder, fname)

def segments_intersect(p1, p2, p3, p4):
    """Return True if 2D segments p1–p2 and p3–p4 intersect."""

    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return (
                min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    return False


def compute_heading(traj):
    """Approximate heading (angle in radians) at segment midpoint."""
    mid = len(traj) // 2
    p_prev, p_next = traj[mid - 1], traj[mid + 1]
    vec = p_next - p_prev
    return np.arctan2(vec[1], vec[0])

# prvious 0.5
def classify_conflict(i, j, means, coeffs, t_vals, dist_thresh=2, turn_thresh=30):
    """
    Classifies conflict between cluster i and j with proper ordering to
    distinguish Turn-across-path from Diverging.
    """
    # Extract 2D projections
    traj1 = means[i, :, :2]
    traj2 = means[j, :, :2]
    c1_x, c1_y, _ = coeffs[i]

    # Compute start/end distance for later merge/diverge
    d_start = point_to_curve_min_dist(c1_x, c1_y, traj2[1, :], t_vals[0], t_vals[-1])
    d_end = point_to_curve_min_dist(c1_x, c1_y, traj2[-2, :], t_vals[0], t_vals[-1])

    # 1) Intersection → Turn-across-path or Crossing
    cross = False
    for a in range(len(traj1) - 2):
        for b in range(len(traj2) - 2):
            if segments_intersect(traj1[a], traj1[a + 1], traj2[b], traj2[b + 1]):
                cross = True
                break
        if cross:
            break

    if cross and (d_start > dist_thresh and d_end > dist_thresh):
        # estimate heading change of j to detect a turn
        h_start = np.arctan2(traj2[3, 1] - traj2[2, 1], traj2[3, 0] - traj2[2, 0])
        h_end = np.arctan2(traj2[-3, 1] - traj2[-4, 1], traj2[-3, 0] - traj2[-4, 0])
        dh = np.degrees((h_end - h_start + np.pi) % (2 * np.pi) - np.pi)
        if abs(dh) > turn_thresh:
            direction = 'left' if dh > 0 else 'right'
            opposite_direction = 'right' if dh > 0 else 'left'
            return f"Turn-{direction}-across-path", f"Turn-{direction}-across-path", f"Turn-{opposite_direction}-across-path", f"Turn-{opposite_direction}-across-path"
        # estimate heading change of i to detect a turn
        h_start = np.arctan2(traj1[3, 1] - traj1[2, 1], traj1[3, 0] - traj1[2, 0])
        h_end = np.arctan2(traj1[-3, 1] - traj1[-4, 1], traj1[-3, 0] - traj1[-4, 0])
        dh = np.degrees((h_end - h_start + np.pi) % (2 * np.pi) - np.pi)
        if abs(dh) > turn_thresh:
            direction = 'left' if dh > 0 else 'right'
            opposite_direction = 'right' if dh > 0 else 'left'
            return f"Turn-{direction}-across-path", f"Turn-{opposite_direction}-across-path", f"Turn-{direction}-across-path", f"Turn-{opposite_direction}-across-path"
        return 'Crossing', 'Crossing', 'Crossing', 'Crossing'

    # 2) Diverging / Merging (only if no intersection)
    if d_start < dist_thresh and d_end > dist_thresh:
        return 'Diverging', 'Head-on', 'Head-on', 'Merging'
    if d_start > dist_thresh and d_end < dist_thresh:
        return 'Merging', 'Head-on', 'Head-on', 'Diverging'

    # 3) Following / Head-on
    dists = [point_to_curve_min_dist(c1_x, c1_y, np.append(pt, 0), t_vals[0], t_vals[-1])
             for pt in traj2]
    mean_dist = np.mean(dists)
    h1 = compute_heading(traj1)
    h2 = compute_heading(traj2)
    ang = np.degrees(abs((h1 - h2 + np.pi) % (2 * np.pi) - np.pi))
    if mean_dist < dist_thresh:
        if ang < 20:
            return 'Following', 'Head-on', 'Head-on', 'Following'
        if ang > 160:
            return 'Head-on', 'Following', 'Following', 'Head-on'

    if abs(d_start-mean_dist) < dist_thresh and abs(d_end-mean_dist) < dist_thresh:
        if ang < 20:
            return 'Following', 'Head-on', 'Head-on', 'Following'
        if ang > 160:
            return 'Head-on', 'Following', 'Following', 'Head-on'

    # 4) Fallback
    return 'Check individual', 'Check individual', 'Check individual', 'Check individual'


def main(config, verbose, config_path):
    tc = TrajectoryClusterer(
        config['cluster_config'],
        cluster_mean_path=config['output_config']['output_folder'],
        meas_id=config['database_config']['measurementID'],
        sens_id=config['database_config']['sensorID'],
        persTrans=None,
        verbose=verbose
    )
    coeffs = tc.get_cluster_mean_coeffs()
    tvals_plot = np.linspace(0, 4, 5)
    means_plot = np.zeros((len(coeffs), len(tvals_plot), 3))
    for i, (cx, cy, cz) in enumerate(coeffs):
        means_plot[i, :, 0] = np.polyval(cx, tvals_plot)
        means_plot[i, :, 1] = np.polyval(cy, tvals_plot)
        means_plot[i, :, 2] = np.polyval(cz, tvals_plot)
    tc.plot_and_save_cluster_means(tvals_plot, means_plot, save_path="cluster_means.png", t_min=-2, t_max=6, n_pts=100,
                                   show_plot=False)

    t_vals = np.linspace(-2, 6, 9)
    means = np.zeros((len(coeffs), len(t_vals), 3))
    for i, (cx, cy, cz) in enumerate(coeffs):
        means[i, :, 0] = np.polyval(cx, t_vals)
        means[i, :, 1] = np.polyval(cy, t_vals)
        means[i, :, 2] = np.polyval(cz, t_vals)

    cols = ['ClusterVehicle1', 'ClusterVehicle2', 'ManeuverClass']
    df = pd.DataFrame(columns=cols)
    n = means.shape[0]

    # ---- START TIMING: maneuver classification ----
    t0 = time.perf_counter()

    for i in range(n):
        for j in range(n):
            if i == j:
                df = df._append({
                    'ClusterVehicle1': i,
                    'ClusterVehicle2': j,
                    'ManeuverClass': 'Following'
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i + len(means),
                    'ClusterVehicle2': j + len(means),
                    'ManeuverClass': 'Following'
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i + len(means),
                    'ClusterVehicle2': j,
                    'ManeuverClass': 'Head-on'
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i,
                    'ClusterVehicle2': j + len(means),
                    'ManeuverClass': 'Head-on'
                }, ignore_index=True)

            else:
                cls_normal, cls_i_opposite, cls_j_opposite, cls_both_opposite = classify_conflict(i, j, means, coeffs,
                                                                                                  t_vals)
                df = df._append({
                    'ClusterVehicle1': i,
                    'ClusterVehicle2': j,
                    'ManeuverClass': cls_normal
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i + len(means),
                    'ClusterVehicle2': j,
                    'ManeuverClass': cls_i_opposite
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i,
                    'ClusterVehicle2': j + len(means),
                    'ManeuverClass': cls_j_opposite
                }, ignore_index=True)
                df = df._append({
                    'ClusterVehicle1': i + len(means),
                    'ClusterVehicle2': j + len(means),
                    'ManeuverClass': cls_both_opposite
                }, ignore_index=True)

    elapsed = time.perf_counter() - t0
    num_maneuvers = len(df)

    # Plot results
    # get all unique maneuver classes
    unique_classes = df['ManeuverClass'].unique()
    # For each maneuver class, plot the corresponding clusters and save the plot
    outputfolder = os.path.join(config['output_config']['output_folder'], 'ManeuverClasses')
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    for cls in unique_classes:
        if cls == 'Check individual':
            continue
        if cls == "Following":
            continue
        cls_df = df[df['ManeuverClass'] == cls]
        for i, j in zip(cls_df['ClusterVehicle1'], cls_df['ClusterVehicle2']):
            if i > j:
                continue
            if i > len(means)-1 or j > len(means)-1:
                continue
            if i > len(means)-1:
                i = i - len(means)
            if j > len(means)-1:
                j = j - len(means)
            plt.figure()
            plt.plot(means[i, :, 0], means[i, :, 1], label=f'Cluster {i}')
            plt.plot(means[j, :, 0], means[j, :, 1], label=f'Cluster {j}')
            plt.title(f'Maneuver Class: {cls}')
            plt.xlabel('X')
            plt.ylabel('Y')
            # plt.legend()
            plt.savefig(os.path.join(outputfolder, f'{cls}_{i}_{j}.png'))
            plt.close()

    # Save df
    meas_id = config['database_config']['measurementID']
    sens_id = config['database_config']['sensorID']
    output_path = os.path.join(config['output_config']['output_folder'], 'ManeuverClasses', f'ManeuverClasses_{meas_id}_{sens_id}.csv')
    df.to_csv(output_path, index=False)
    print(f'Saved maneuver classes to {output_path}')


    # Write timing CSV
    timing_csv_path = build_time_csv_path(config_path, config['output_config']['output_folder'])
    os.makedirs(os.path.dirname(timing_csv_path), exist_ok=True)
    with open(timing_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Number of maneuver", "Time"])
        writer.writerow([num_maneuvers, f"{elapsed:.6f}"])
    print(f"Maneuver timing CSV written to: {timing_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')

    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.verbose, args.config)
