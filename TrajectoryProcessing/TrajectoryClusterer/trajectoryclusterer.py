"""
Created on Feb 11 2025 11:27

@author: ISAC - pettirsch
"""
import os.path
import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

class TrajectoryClusterer:
    def __init__(self, cluster_config, cluster_mean_path, meas_id=None, sens_id=None, persTrans=None, verbose=False):

        self.cluster_algorithm = cluster_config['cluster_algorithm']
        self.cost_threshold = cluster_config['cost_threshold']
        self.class_wise = cluster_config['class_wise']
        self.cost_feature = cluster_config['cost_feature']
        self.timing = {"clustering": 0.0, "assign_cluster": 0.0}

        self.cluster_mean_path = cluster_mean_path
        self.measID = meas_id
        self.sensID = sens_id

        self.persTrans = persTrans

        self.load_cluster_means()

        self.verbose = verbose

    def load_cluster_means(self):

        # Save the coefficients to a file
        inputpath = os.path.join(self.cluster_mean_path, "Cluster",
                                 "measID_{}_sensID_{}_cluster_mean_coeffs.npy".format(self.measID, self.sensID))

        if os.path.exists(inputpath):
            self.mean_cluster_coeffs = np.load(inputpath, allow_pickle=True).tolist()
            print("Cluster means loaded from file.")
        else:
            self.mean_cluster_coeffs = []
            print("No cluster means found. Starting with empty list.")

    def set_frame(self, frame):
        """
        Set the frame for visualization.
        """
        self.frame = frame

    def cluster_trajectories(self, trajectories, frame):


        self.frame = frame

        if self.cost_feature == 'Start_End_Distance':
            self.cluster_trajectories_start_end_distance(trajectories)
            num_cluster = 50
        elif self.cost_feature == "grid":
            self.cluster_trajectories_grid(trajectories)
            num_cluster = 50
        elif self.cost_feature == "key_points":
            num_cluster = self.cluster_trajectories_key_points(trajectories)
        elif self.cost_feature == "Heading":
            self.cluster_trajectories_heading(trajectories)
            num_cluster = 50

        return num_cluster, dict(self.timing)


    def downsample_by_distance(self, traj, num_points=5):
        """
        Downsample trajectory based on % of total distance.
        traj: (N, 2) array of x, y
        Returns: (num_points * 2,) flattened array
        """
        traj_2d = traj[:, :2]  # Ensure we only use x, y coordinates
        traj_3d = traj[:, :3]  # Keep the original 3D coordinates if needed

        # Compute cumulative distance along the trajectory
        diffs = np.diff(traj_2d, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cumdist = np.insert(np.cumsum(dists), 0, 0)

        # Normalize to 0-1
        cumdist /= cumdist[-1]

        # Target distances: evenly spaced
        target_cumdist = np.linspace(0, 1, num_points)

        # Interpolate
        new_points = []
        for t in target_cumdist:
            idx = np.searchsorted(cumdist, t)
            if idx >= len(traj):
                idx = len(traj) - 1
            new_points.append(traj_3d[idx])

        return np.array(new_points)

    def plot_and_save_cluster_means(
            self,
            t_original,
            means,
            save_path="cluster_means.svg",
            t_min=-10,
            t_max=10,
            n_pts=200,
            show_plot=True
    ):

        import matplotlib as mpl
        """
        Fit 4th-degree polynomials to each cluster mean (in 3D) and
        save both a matplotlib plot and a “thermal” image overlay
        with matching colors.
        """
        C, T, D = means.shape
        assert D == 3, "Expected 3D trajectories (x,y,z)"

        # Use distinct RGB colors
        bgr_colors = generate_distinct_bgr(C)
        #rgb_colors = generate_distinct_rgb(C)
        rgb_colors = [(r, g, b) for (b, g, r) in bgr_colors]
        mpl_colors = [(r / 255, g / 255, b / 255) for (r, g, b) in rgb_colors]

        # Prepare time points for plotting
        t_plot = np.linspace(t_min, t_max, n_pts)

        # --- Matplotlib plot ---
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
        })

        fig, ax = plt.subplots(figsize=(12 / 2.54, 16/ 2.54))  # 8cm × 5cm

        # Plot each cluster
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for i in range(C):
            coeffs_x = np.polyfit(t_original, means[i, :, 0], 4)
            coeffs_y = np.polyfit(t_original, means[i, :, 1], 4)
            x_plot = np.polyval(coeffs_x, t_plot)
            y_plot = np.polyval(coeffs_y, t_plot)
            ax.plot(x_plot, y_plot, color=mpl_colors[i], label=f"Cluster {i}")
            xmin = min(xmin, x_plot.min())
            xmax = max(xmax, x_plot.max())
            ymin = min(ymin, y_plot.min())
            ymax = max(ymax, y_plot.max())

        x_diff = xmax - xmin
        y_diff = ymax - ymin
        # Plot equal range -> set limits
        if x_diff > y_diff:
            ax.set_xlim(xmin, xmin + x_diff)
            ax.set_ylim(ymin, ymin + x_diff)
        else:
            ax.set_ylim(ymin, ymin + y_diff)
            ax.set_xlim(xmin, xmin + y_diff)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, linestyle=":", linewidth=0.5)

        # # Add 2-row legend below plot
        # legend = ax.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, -0.4),
        #     ncol=2,  # e.g., 2 rows of 6 if you have 12 clusters
        #     frameon=False,
        #     handletextpad=0.5,
        #     columnspacing=1.0,
        #     borderaxespad=0
        # )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.4)  # make space manually for legend
        # legend = ax.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, -0.15),
        #     bbox_transform=fig.transFigure,  # << Key change
        #     ncol=2,
        #     frameon=False,
        #     fontsize=11
        # )
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,  # e.g., 2 rows of 6 if you have 12 clusters
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0
        )

        # fig.subplots_adjust(bottom=0.15)  # space for legend
        # Ensure tight layout
        # fig.tight_layout()

        # SEt axis 1
        # ax.set_box_aspect(1)

        # Save
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="svg", dpi=300)
        plt.close()
        print(f"Saved plot as '{save_path}'")

        # --- Thermal / frame overlay ---
        if self.persTrans is not None and hasattr(self, "frame") and self.frame is not None:
            world_points = []
            for i in range(C):
                coeffs_x = np.polyfit(t_original, means[i, :, 0], 4)
                coeffs_y = np.polyfit(t_original, means[i, :, 1], 4)
                coeffs_z = np.polyfit(t_original, means[i, :, 2], 4)
                xyz = np.vstack([
                    np.polyval(coeffs_x, t_plot),
                    np.polyval(coeffs_y, t_plot),
                    np.polyval(coeffs_z, t_plot)
                ]).T  # shape (n_pts, 3)
                world_points.append(xyz)

            frame_overlay = self.frame.copy()
            for i, pts in enumerate(world_points):
                pixel_pts = self.persTrans.worldToPixel(pts)
                pixel_pts = np.round(pixel_pts).astype(np.int32)

                h, w = frame_overlay.shape[:2]
                valid = (
                        (pixel_pts[:, 0] >= 0) & (pixel_pts[:, 0] < w) &
                        (pixel_pts[:, 1] >= 0) & (pixel_pts[:, 1] < h)
                )
                pixel_pts = pixel_pts[valid]

                if len(pixel_pts) > 1:
                    cv2.polylines(
                        frame_overlay,
                        [pixel_pts],
                        isClosed=False,
                        color=bgr_colors[i],
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

            overlay_path = os.path.splitext(save_path)[0] + "_overlay.png"
            cv2.imwrite(overlay_path, frame_overlay)
            print(f"✅ Saved thermal overlay as '{overlay_path}'")


    def cluster_trajectories_key_points(self, trajectories):

        # reset timings for this run
        self.timing = {"clustering": 0.0, "assign_cluster": 0.0}

        # Create clusters if they were not loaded
        if len(self.mean_cluster_coeffs) == 0:
            clustering_start = time.perf_counter()

            # Print Number of trajectories
            print("Number of trajectories: ", len(trajectories))

            # Extract key points from trajectories
            world_points = [traj.get_world_positions() for traj in trajectories]

            # Create feature vectors
            feature_vectors_all = np.array([self.downsample_by_distance(coords) for coords in world_points])
            feature_vectors = feature_vectors_all[:, :, :2]
            N, K, D = feature_vectors.shape

            # Calc pointwise distance
            dist = np.zeros((N, N, K), dtype=np.float32)
            for k in range(K):
                # compute the N×N distances at key‐point index k
                dist[:, :, k] = cdist(
                    feature_vectors[:, k, :],  # all trajectories’ k-th point
                    feature_vectors[:, k, :],  # against itself
                    metric='euclidean'
                )

            # Use max distance of the 5 points to calculate the cost
            cost_max = dist.max(axis=2)

            # Cluster using DBSCAN with eps: 0.5 and min_samples=5
            dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')  # 0.5 und 10 vorher
            dbscan_labels = dbscan.fit_predict(cost_max)

            # Get unique labels
            unique_labels = np.unique(dbscan_labels)
            valid_mask = unique_labels != -1
            valid_labels = unique_labels[valid_mask]

            # Print numbers of all trajectories and number of trajectory with label -1
            print("Number of clusters after DBSCAN: ", len(valid_labels))
            print("Number of not assigned trajectories after DBSCAN: ", np.sum(dbscan_labels == -1))

            # Step 1: Compute cluster means, skipping noise (-1)
            # Create a dictionary to store means by label
            cluster_means_all = np.stack([
                feature_vectors_all[dbscan_labels == label].mean(axis=0)
                for label in valid_labels
            ])
            cluster_means = cluster_means_all[:, :, :2]  # Keep only x and y coordinates

            # Shape: (C, 5, D)
            means_all = cluster_means_all
            means = means_all[:, :, :2]  # Keep only x and y coordinates

            # means = cluster_means
            C, T, D = means.shape  # T = 30

            diffs_mean_top_k = np.zeros((C, C))

            t0_merge = -10
            t1_merge = 14
            t_vals = np.linspace(0, 4, means.shape[1])  # To get 0, 25%, 50%, 75% and 1

            # Save weights
            weights = []
            for i in range(C):
                weights.append(np.sum(dbscan_labels == i))

            clusters_curr = dbscan_labels

            # --- merge‐loop ---
            finish_summary = False
            while not finish_summary:
                C, T, D = means.shape

                # 1) fit parametric polys for each mean‐trajectory
                coeffs_x = np.stack([np.polyfit(t_vals, mean[:, 0], 2) for mean in means])
                coeffs_y = np.stack([np.polyfit(t_vals, mean[:, 1], 2) for mean in means])

                # 2) compute pairwise distance matrix by summing all pt→curve distances
                diffs = np.full((C, C), np.inf, dtype=np.float32)
                for i in range(C):

                    for j in range(C):

                        # Don't compare to self
                        if i == j:
                            continue

                        # 3) sum distances from every point in i→curve_j and j→curve_i
                        distances = []
                        # points of cluster j
                        for pt in means[j]:
                            distances.append(point_to_curve_min_dist(
                                coeffs_x[i], coeffs_y[i], pt, t0_merge, t1_merge
                            ))
                        total_distance = np.sum(distances)

                        # only keep if below your merge threshold
                        if max(distances) < 1 or total_distance < 5:
                            diffs[i, j] = max(distances)

                # 4) pick the best pair to merge
                ij = np.unravel_index(np.argmin(diffs), diffs.shape)
                best_dist = diffs[ij]
                if best_dist == np.inf:
                    finish_summary = True
                    break

                i, j = ij

                # 5) merge means i & j (here equally weighted; you could weight by cluster size)
                weight_i = weights[i]
                weight_j = weights[j]
                new_mean_all = (means_all[i] * weight_i + means_all[j] * weight_j) / (weight_i + weight_j)
                new_mean = new_mean_all[:, :2]  # Keep only x and y coordinates

                # 6) build new means list: drop i,j, append new_mean
                keep = [k for k in range(C) if k not in (i, j)]
                weights = [weights[k] for k in keep]
                means_all = np.vstack([means_all[k][None] for k in keep] + [new_mean_all[None]])
                means = np.vstack([means[k][None] for k in keep] + [new_mean[None]])
                weights.append(weight_i + weight_j)


            # finally, re‐fit and save your cluster‐mean polynomials as before
            coeffs_x, coeffs_y, coeffs_z = [], [], []
            t_vals_new = np.linspace(0, 4, means_all.shape[1])
            for mean_traj in means_all:
                if np.linalg.norm(mean_traj[0, :2] - mean_traj[-1, :2]) < 3:
                    continue
                coeffs_x.append(np.polyfit(t_vals_new, mean_traj[:, 0], 2))
                coeffs_y.append(np.polyfit(t_vals_new, mean_traj[:, 1], 2))
                coeffs_z.append(np.polyfit(t_vals_new, mean_traj[:, 2], 2))

            self.mean_cluster_coeffs = list(zip(coeffs_x, coeffs_y, coeffs_z))
            self.save_cluster_mean_coeffs()

            self.timing["clustering"] = time.perf_counter() - clustering_start

            print("number of cluster: ", len(self.mean_cluster_coeffs))
        else:
            self.timing["clustering"] = 0.0

        # Step 4: Assign new labels
        assign_start = time.perf_counter()
        print("Number of clusters: ", len(self.mean_cluster_coeffs))

        new_labels = -1 * np.ones(len(trajectories), dtype=int)
        t_vals = np.linspace(0, 4, 5)
        free_cluster_ids = [i + len(self.mean_cluster_coeffs) for i in range(len(self.mean_cluster_coeffs))]

        # Calc means based on self.mean_cluster_coeffs
        means_all = np.zeros((len(self.mean_cluster_coeffs), 5, 3))
        for i, (coeffs_x, coeffs_y, coeffs_z) in enumerate(self.mean_cluster_coeffs):
            means_all[i, :, 0] = np.polyval(coeffs_x, t_vals)
            means_all[i, :, 1] = np.polyval(coeffs_y, t_vals)
            means_all[i, :, 2] = np.polyval(coeffs_z, t_vals)

        self.plot_and_save_cluster_means(t_vals, means_all, save_path="cluster_means.png", t_min=-2, t_max=6, n_pts=100,
                                         show_plot=False)

        t0 = -2
        t1 = 6

        means = means_all[:, :, :2]  # Keep only x and y coordinates
        print("Start assigning cluster to trajectories")

        for idx, traj in enumerate(trajectories):
            # 1) downsample and fit parametric poly to this trajectory
            traj_pts = self.downsample_by_distance(traj.get_world_positions(), num_points=5)  # (5,2)
            traj_pts = traj_pts[:, :2]  # Ensure we only use x, y coordinates

            min_distance = np.inf
            max_all_distance = np.inf
            best_cluster = -1

            # 2) compare against each cluster’s polynomial
            for cluster_idx, (coeffs_x_cl, coeffs_y_cl, coeffs_z_cl) in enumerate(self.mean_cluster_coeffs):
                # direction check
                p0_cl = np.array([np.polyval(coeffs_x_cl, 0),
                                  np.polyval(coeffs_y_cl, 0)])
                p1_cl = np.array([np.polyval(coeffs_x_cl, 4),
                                  np.polyval(coeffs_y_cl, 4)])
                # vec_cl = p1_cl - p0_cl

                # 3) for each of the 5 pts, compute min distance to this cluster‐curve
                distances = []
                for pt in traj_pts:
                    distances.append(point_to_curve_min_dist(
                        coeffs_x_cl, coeffs_y_cl, pt, t0, t1
                    ))
                total_distance = np.sum(distances)
                max_distance = np.max(distances)

                # track best match
                if total_distance < min_distance:
                    # if total_distance < min_distance:
                    min_distance = total_distance
                    max_all_distance = max_distance
                    distance_start = np.linalg.norm(p0_cl - traj_pts[0])
                    distance_end = np.linalg.norm(p0_cl - traj_pts[4])

                    if distance_end < distance_start:
                        best_cluster = free_cluster_ids[cluster_idx]
                    else:
                        best_cluster = cluster_idx

            # 4) threshold and assign
            if min_distance < 10 or max_all_distance <= 3:
            #if min_distance < 10 and max_all_distance < 2:  # whatever threshold you choose
                new_labels[idx] = best_cluster
            else:
                new_labels[idx] = -1

            traj.set_cluster_label(str(new_labels[idx]))

        valid_labels = np.unique(new_labels[new_labels != -1])

        self.timing["assign_cluster"] = time.perf_counter() - assign_start

        # `new_labels` now contains updated cluster assignments
        print("Updated cluster labels assigned.")
        print("Number of final clusters:", len(valid_labels))
        print("Number still unassigned:", np.sum(new_labels == -1))

        return len(self.mean_cluster_coeffs)*2


    def save_cluster_mean_coeffs(self):

        # Save the coefficients to a file
        coeffs = np.array(self.mean_cluster_coeffs)
        outputpath = os.path.join(self.cluster_mean_path, "Cluster",
                                  "measID_{}_sensID_{}_cluster_mean_coeffs.npy".format(self.measID, self.sensID))
        np.save(outputpath, coeffs)
        print("Cluster mean coefficients saved to:", outputpath)

    def get_cluster_mean_coeffs(self):
        # Load the coefficients from a file
        return self.mean_cluster_coeffs


def point_to_curve_min_dist(coeffs_x, coeffs_y, pt, t0, t1):
    px = np.poly1d(coeffs_x)
    py = np.poly1d(coeffs_y)

    # build D(t)
    dx = px - pt[0]
    dy = py - pt[1]
    D = np.polyadd(np.polymul(dx, dx), np.polymul(dy, dy))
    Dp = D.deriv()

    # find real stationary points in [t0,t1]
    roots = np.roots(Dp)
    ts = roots[np.isreal(roots)].real
    ts = ts[(ts >= t0) & (ts <= t1)]

    # add endpoints, dedupe
    candidates = np.unique(np.hstack([ts, t0, t1]))

    # evaluate & clamp
    Dvals = D(candidates)
    minD = max(Dvals.min(), 0)

    # optionally also return t_min:
    # t_min = candidates[Dvals.argmin()]

    return np.sqrt(minD)

def rgb_dist(c1, c2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(c1, c2)))

def generate_distinct_bgr(n, candidates=1000):
    # Start with one random color (RGB in [0,1])
    chosen = [(random.random(), random.random(), random.random())]

    while len(chosen) < n:
        # sample a bunch of random candidates
        pool = [(random.random(), random.random(), random.random()) for _ in range(candidates)]
        # for each, find its distance to the nearest already-chosen color
        best_c, best_d = None, -1
        for p in pool:
            d = min(rgb_dist(p, c) for c in chosen)
            if d > best_d:
                best_d, best_c = d, p

        chosen.append(best_c)

    # convert to 0–255 BGR tuples
    bgr = [
        (int(b*255), int(g*255), int(r*255))
        for (r, g, b) in chosen
    ]
    return bgr


def rgb_dist(c1, c2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2)))

def generate_distinct_rgb(n, candidates=1000):
    # Start with one random color (RGB in [0,1])
    chosen = [(random.random(), random.random(), random.random())]

    while len(chosen) < n:
        # sample a bunch of random candidates
        pool = [(random.random(), random.random(), random.random()) for _ in range(candidates)]
        # for each, find its distance to the nearest already-chosen color
        best_c, best_d = None, -1
        for p in pool:
            d = min(rgb_dist(p, c) for c in chosen)
            if d > best_d:
                best_d, best_c = d, p

        chosen.append(best_c)

    # convert to 0–255 RGB tuples
    rgb = [
        (int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b) in chosen
    ]
    return rgb
