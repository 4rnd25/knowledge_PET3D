"""
Created on Dec 11 09:10

@author: ISAC - pettirsch
"""

import argparse
import numpy as np
import cv2

from CommonUtils.PerspectiveTransform.perspectiveTransform import PerspectiveTransform
from scipy.spatial import Delaunay
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument('--Faces', type=str,
                    default='/data/Arnd/paper_monocular_3D_object_detection/New_Experiments/Infrastructure/Clicked_surface_multiple_points/12004/roadsurfaceTriangulationFaces.csv',
                    help='json config path')
parser.add_argument('--Points', type=str,
                    default='/data/Arnd/paper_monocular_3D_object_detection/New_Experiments/Infrastructure/Clicked_surface_multiple_points/12004/roadsurfaceTriangulationPoints.csv',
                    help='json config path')
parser.add_argument('--calibrationPath', type=str,
                    default='/data/Arnd/paper_monocular_3D_object_detection/New_Experiments/Infrastructure/GT_Data/12004/calibrationMatrix.xml',
                    help='json config path')
parser.add_argument('--outputpath',
                    default='/data/Arnd/paper_monocular_3D_object_detection/New_Experiments/Infrastructure/Plot_sparse_triangulation/',
                    help='save to project/name')
parser.add_argument("--filePath", help="path to video or image file",
                    default="/data/Arnd/paper_monocular_3D_object_detection/New_Experiments/Infrastructure/Images/12004.png")
opt = parser.parse_args()

def plot_triangulation(path_faces, path_points, calibrationPath, output_path,
                       color=(87, 171, 39), imageFile=None):

    # Create Perspective Transform
    perspectiveTransform = PerspectiveTransform()

    # Update perspective transform
    perspectiveTransform.updateCalibration(calibrationPath=calibrationPath,
                                           triangulationFacesPath=path_faces,
                                           triangulationPointsPath=path_points,
                                           calibration_type="Delaunay triangulation")

    # Load faces from csv
    faces = np.loadtxt(path_faces, delimiter=',')
    points = np.loadtxt(path_points, delimiter=',')

    def restrict_triangulation_to_street(points, connectivity_tolerance=1.5):
        """
        Restrict triangulation to avoid triangles that cross the street curve. Just for plotting

        Parameters:
            points (numpy.ndarray): Array of shape (n, 2) or (n, 3), representing street points.
            connectivity_tolerance (float): Maximum distance factor to allow connections.

        Returns:
            tuple: (filtered_points, valid_triangles)
        """
        # Step 1: Perform Delaunay triangulation in 2D (ignore z)
        points_xy = points[:, :2]  # Use only x, y for triangulation
        delaunay = Delaunay(points_xy)
        triangles = np.array(delaunay.simplices, dtype=np.int32)

        # Step 2: Filter triangles that bypass the curve
        def is_valid_triangle(triangle):
            # Extract the points of the triangle
            p1, p2, p3 = points_xy[triangle]
            d12 = np.linalg.norm(p1 - p2)
            d23 = np.linalg.norm(p2 - p3)
            d31 = np.linalg.norm(p3 - p1)
            # Compute pairwise distances between sequential points
            distances = np.array([d12, d23, d31])
            max_distance = connectivity_tolerance * np.mean(distances)

            if np.mean(distances) > 5:
                return False
            # Ensure distances are not too large (avoids cross-curve triangles)
            return np.all(distances < max_distance)

        valid_triangles = [triangle for triangle in triangles if is_valid_triangle(triangle)]
        valid_triangles = np.array(valid_triangles, dtype=np.int32)

        return points, valid_triangles

    # points, faces = restrict_triangulation_to_street(points)

    # Project Points to image
    projected_points = perspectiveTransform.worldToPixel(points)

    # Load image
    image = cv2.imread(imageFile)

    # Create a copy of the image to draw lines on
    overlay = image.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through all faces
    for face in faces:

        pt_1 = projected_points[int(face[0])]
        pt_2 = projected_points[int(face[1])]
        pt_3 = projected_points[int(face[2])]

        pt_1 = np.round(pt_1).astype(int)
        pt_2 = np.round(pt_2).astype(int)
        pt_3 = np.round(pt_3).astype(int)

        pt_1_3d = points[int(face[0])]
        pt_2_3d = points[int(face[1])]
        pt_3_3d = points[int(face[2])]

        if pt_1[0] < 0 and pt_1[1] < 0:
            continue
        if pt_2[0] < 0 and pt_2[1] < 0:
            continue
        if pt_3[0] < 0 and pt_3[1] < 0:
            continue

        # # Check if all face points are in the image
        num_out = 0
        if (pt_1[0] < 5 or pt_1[0] >= image.shape[1] or pt_1[1] < 0 or pt_1[1] >= image.shape[0]):
            num_out += 1

        if (pt_2[0] < 5 or pt_2[0] >= image.shape[1] or pt_2[1] < 0 or pt_2[1] >= image.shape[0]):
            num_out += 1
        if (pt_3[0] < 5 or pt_3[0] >= image.shape[1] or pt_3[1] < 0 or pt_3[1] >= image.shape[0]):
            num_out += 1

        # 71
        # if (pt_1[0] < 0 and pt_1[1] > 200) or (pt_2[0] < 0 and pt_2[1] > 200) or (pt_3[0] < 0 and pt_3[1] > 200):
        #     continue
        # if (pt_1[0] < 30 and pt_1[1] > 300) or (pt_2[0] < 30 and pt_2[1] > 300) or (pt_3[0] < 30 and pt_3[1] > 300):
        #     continue
        # if pt_1[0] > 630 or pt_2[0] > 630 or pt_3[0] > 630:
        #     if (pt_1[1] >255 and pt_1[1] < 350) or (pt_2[1] > 255 and pt_2[1] < 350) or (pt_3[1] > 255 and pt_3[1] < 350):
        #         continue

        if (pt_1[0] < 0 or pt_1[0] >= image.shape[1]) and (pt_1[1] < 0 or pt_1[1] >= image.shape[0]):
            continue
        if (pt_2[0] < 0 or pt_2[0] >= image.shape[1]) and (pt_2[1] < 0 or pt_2[1] >= image.shape[0]):
            continue
        if (pt_3[0] < 0 or pt_3[0] >= image.shape[1]) and (pt_3[1] < 0 or pt_3[1] >= image.shape[0]):
            continue
        if num_out > 2:
            continue

        pt_1[0] = np.clip(pt_1[0], 0, image.shape[1] - 1)
        pt_1[1] = np.clip(pt_1[1], 0, image.shape[0] - 1)
        pt_2[0] = np.clip(pt_2[0], 0, image.shape[1] - 1)
        pt_2[1] = np.clip(pt_2[1], 0, image.shape[0] - 1)
        pt_3[0] = np.clip(pt_3[0], 0, image.shape[1] - 1)
        pt_3[1] = np.clip(pt_3[1], 0, image.shape[0] - 1)

        # cv2 fillPoly
        cv2.fillPoly(overlay, [np.array([pt_1, pt_2, pt_3])], (255,255,255))

        # Draw the face on the image
        cv2.line(overlay, (int(pt_1[0]), int(pt_1[1])), (int(pt_2[0]), int(pt_2[1])), color, 2)
        cv2.line(overlay, (int(pt_2[0]), int(pt_2[1])), (int(pt_3[0]), int(pt_3[1])), color, 2)
        cv2.line(overlay, (int(pt_3[0]), int(pt_3[1])), (int(pt_1[0]), int(pt_1[1])), color, 2)

        # Plot circle at each point
        cv2.circle(overlay, (int(pt_1[0]), int(pt_1[1])), 4, (30,7,204), -1)
        cv2.circle(overlay, (int(pt_2[0]), int(pt_2[1])), 4, (30,7,204), -1)
        cv2.circle(overlay, (int(pt_3[0]), int(pt_3[1])), 4, (30,7,204), -1)

        ax.plot([pt_1_3d[0], pt_2_3d[0]], [pt_1_3d[1], pt_2_3d[1]], [pt_1_3d[2], pt_2_3d[2]], color='b')
        ax.plot([pt_2_3d[0], pt_3_3d[0]], [pt_2_3d[1], pt_3_3d[1]], [pt_2_3d[2], pt_3_3d[2]], color='b')
        ax.plot([pt_3_3d[0], pt_1_3d[0]], [pt_3_3d[1], pt_1_3d[1]], [pt_3_3d[2], pt_1_3d[2]], color='b')

    # Save the plot as image
    # plt.savefig(output_path + 'triangulation_3d_71.png')

    # Blend the overlay with the original image
    alpha = 0.7  # Transparency factor
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Save the image
    cv2.imwrite(output_path + 'triangulation_sparse_12004.png', image)

if __name__ == '__main__':
    plot_triangulation(path_faces=opt.Faces, path_points=opt.Points, calibrationPath=opt.calibrationPath,
                            output_path=opt.outputpath, imageFile = opt.filePath)