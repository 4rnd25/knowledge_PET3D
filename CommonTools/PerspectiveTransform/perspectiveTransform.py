"""
Created on 03 05 2024 07:31

@author: ISAC - pettirsch
"""

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re

from sklearn.neighbors import NearestNeighbors

from CommonUtils.Vector_Utils.gram_schmidt_process import gram_schmidt
from CommonUtils.Vector_Utils.normal_vector_alignment import allign_normal_vector_for_hnf

class PerspectiveTransform:

    def __init__(self, calibrationPath=None, triangulationFacesPath=None, triangulationPointsPath=None,
                 calibration_type="Delaunay triangulation", imageSize = (640,480), create_bottom_map = True, verbose=True):

        # Init image size
        self.imageSize = imageSize
        self._verbose = verbose
        self.bottom_map = None
        self.create_bottom_map = create_bottom_map

        self._homography = None

        # Default constructor logic
        if calibrationPath is None and triangulationFacesPath is None and triangulationPointsPath is None:
            self.calibrationPath = ""
            self.triangulationFacesPath = ""
            self.triangulationPointsPath = ""
            self.calibration_type = None

        else:
            # Ensure required parameters are provided
            assert calibrationPath is not None, "calibrationPath is None"
            assert triangulationFacesPath is not None, "triangulationFacesPath is None"
            assert triangulationPointsPath is not None, "triangulationPointsPath is None"
            assert calibration_type == "Delaunay triangulation"

            self.calibrationPath = ""
            self.triangulationFacesPath = ""
            self.triangulationPointsPath = ""

            # Update internal calibration data
            self.loadCalibrationFromXML(calibrationPath)

            # Update faces and points
            self.loadTriangulationFacesFromCSV(triangulationFacesPath)
            self.loadTriangulationPointsFromCSV(triangulationPointsPath)

            # Save the path
            self.calibrationPath = calibrationPath
            self.triangulationFacesPath = triangulationFacesPath
            self.triangulationPointsPath = triangulationPointsPath
            self.calibration_type = calibration_type


            # Init calibration
            self.initCalibration()

    def updateCalibration(self, calibrationPath= None, triangulationFacesPath=None, triangulationPointsPath = None,
                          homographyPath= None,
                          calibration_type = None):

        # Assert internal calibration data is not None
        self.calibrationPath = calibrationPath
        assert self.calibrationPath is not None, "calibrationPath is None"

        # Update internal calibration data
        self.loadCalibrationFromXML(calibrationPath)

        if calibration_type == "Delaunay triangulation":
            assert triangulationFacesPath is not None, "triangulationFacesPath is None"
            assert triangulationPointsPath is not None, "triangulationPointsPath is None"

            # Update faces and points
            self.loadTriangulationFacesFromCSV(triangulationFacesPath)
            self.loadTriangulationPointsFromCSV(triangulationPointsPath)

            self.calibration_type = calibration_type
        elif calibration_type == "Homography":
            assert homographyPath is not None, "homographyPath is None"

            self._homography = np.loadtxt(homographyPath, delimiter=",")
            self.calibration_type = calibration_type
        elif calibration_type == "Projection_only":
            self.calibration_type = calibration_type
        else:
            raise NotImplementedError("Calibration type {} is not supported".format(calibration_type))

        # Init calibration
        self.initCalibration()
        if self._verbose:
            print("Calibration updated")


    def loadCalibrationFromXML(self, path):
        """
        Load the calibration data from the xml file.
        :param path: Path to the xml file
        :return: cailibration_type, cameraPosition, cameraRotMat, cameraTransVec, cameraMat, distortionCoefficients
        """

        if path is None:
            raise ValueError("Path is None")

        if path == self.calibrationPath:
            self.updatedCalibration = False

        tree = ET.parse(path)
        root = tree.getroot()

        camera_position_str = root.find('Camera_Position').text
        camera_position = np.array(re.findall(r'-?\d+\.\d+', camera_position_str), dtype=float)

        camera_rotation_matrix_str = root.find('Camera_Rotationmatrix').text
        camera_rotation_matrix = np.array(re.findall(r'-?\d+\.\d+', camera_rotation_matrix_str), dtype=float)
        camera_rotation_matrix = camera_rotation_matrix.reshape(3, 3)

        camera_translation_vector_str = root.find('Camera_Translation_Vector').text
        camera_translation_vector = np.array(re.findall(r'-?\d+\.\d+|\d+', camera_translation_vector_str), dtype=float)

        camera_matrix_str = root.find('Camera_Matrix').text
        camera_matrix_values = re.findall(r'[+-]?\d+(?:\.\d+)?', camera_matrix_str)
        camera_matrix = np.array(camera_matrix_values, dtype=float).reshape(3, 3)

        distortion_coefficients_str = root.find('Distortion_Coefficients').text
        distortion_coefficients_values = re.findall(r'[+-]?\d+(?:\.\d+)?', distortion_coefficients_str)
        distortion_coefficients = np.array(distortion_coefficients_values, dtype=float)

        self.calibrationPath = path
        self.updatedCalibration = True

        self.cameraPosition = camera_position
        self.cameraRotMat = camera_rotation_matrix
        self.cameraTransVec = camera_translation_vector
        self.cameraMat = camera_matrix
        self.distortionCoefficients = distortion_coefficients
        self.cameraRT = np.eye(4)
        self.cameraRT[:3, :3] = self.cameraRotMat
        self.cameraRT[:3, 3] = self.cameraTransVec
        self.cameraRTInv = np.linalg.inv(self.cameraRT)


    def initCalibration(self):
        """
        Initialize the calibration data.
        """

        self.cameraRotVec = cv2.Rodrigues(self.cameraRotMat)[0]
        self.cameraMatInv = np.linalg.inv(self.cameraMat)
        transformationMat = np.zeros((3, 4))
        transformationMat[0:3, 0:3] = self.cameraRotMat
        transformationMat[0:3, 3] = self.cameraTransVec
        self.projectionMatrix = np.dot(self.cameraMat, transformationMat)
        self.projectionMatrix03Inv = np.linalg.inv(self.projectionMatrix[0:3, 0:3])
        self.cameraPosition = np.dot(-self.cameraRotMat.T, self.cameraTransVec)


        if self.calibration_type == "Delaunay triangulation":
            self._initDelaunayTriangulation()
            if self.create_bottom_map:
                self.bottom_map = self.createBottomMap()
            self.updatedCalibration = False
            self.updatedTriangulationPoints = False
            self.updatedTriangulationFaces = False
        elif self.calibration_type == "Homography":
            self._updatedCalibration = False
        elif self.calibration_type == "Projection_only":
            self.updatedCalibration = False
        else:
            raise NotImplementedError("Calibration type {} is not supported".format(self.cailibration_type))

    def loadTriangulationFacesFromCSV(self, path):
        """
        Load the triangulation faces from the csv file.
        :param path: Path to the csv file
        :return: triangulationFaces
        """

        if path is None:
            return None

        triangulationFaces = np.loadtxt(path, delimiter=",", dtype=np.int32)

        self.triangulationFacesPath = path
        self.triangulationFaces = triangulationFaces
        self.updatedTriangulationFaces = True


    def loadTriangulationPointsFromCSV(self, path):
        """
        Load the triangulation points from the csv file.
        :param path: Path to the csv file
        :return: triangulationPoints
        """

        if path is None:
            raise ValueError("Path is None")
            # return None

        triangulationPoints = np.loadtxt(path, delimiter=",", dtype=np.float64)

        self.triangulationPointsPath = path
        self.triangulationPoints = triangulationPoints
        self.updatedTriangulationPoints = True

    def _initDelaunayTriangulation(self, plot_look_up_matrix=False):
        """
        Initialize the Delaunay triangulation.
        """

        assert self.triangulationFaces is not None, "Triangulation faces are None"
        assert self.triangulationPoints is not None, "Triangulation points are None"


        triangulationPointsPixel = self.worldToPixel(self.triangulationPoints)

        self.LookUpMatrix = np.ones((self.imageSize[1], self.imageSize[0], 1), dtype='int32')
        self.LookUpMatrix[:, :, 0] = -1

        for i in range(self.triangulationFaces.shape[0]):
            pt_1 = triangulationPointsPixel[self.triangulationFaces[i][0]]
            pt_2 = triangulationPointsPixel[self.triangulationFaces[i][1]]
            pt_3 = triangulationPointsPixel[self.triangulationFaces[i][2]]

            # Check if the points are in the image
            if ((pt_1[0] < 0 or pt_1[0] >= self.imageSize[0]) or (pt_1[1] < 0 or pt_1[1] >= self.imageSize[1])) and \
                    ((pt_2[0] < 0 or pt_2[0] >= self.imageSize[0]) or (
                            pt_2[1] < 0 or pt_2[1] >= self.imageSize[1])) and \
                    ((pt_3[0] < 0 or pt_3[0] >= self.imageSize[0]) or (pt_3[1] < 0 or pt_3[1] >= self.imageSize[1])):
                continue

            cv2.fillConvexPoly(self.LookUpMatrix, np.array([pt_1, pt_2, pt_3], dtype='int32'), color=int(i))

        lookup_matrix_squeezed = np.squeeze(self.LookUpMatrix.copy())
        minus_one_coords = np.argwhere(lookup_matrix_squeezed == -1)
        pos_coords = np.argwhere(lookup_matrix_squeezed != -1)
        tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pos_coords)
        distances, indices = tree.kneighbors(minus_one_coords, return_distance=True)

        valid_indices = np.where(distances < 1000000)[0]
        filtered_indices = indices[valid_indices,0]
        filtered_minus_one_coords = minus_one_coords[valid_indices,:]

        self.LookUpMatrix[filtered_minus_one_coords[:, 0], filtered_minus_one_coords[:, 1], 0] = self.LookUpMatrix[
           pos_coords[filtered_indices, 0], pos_coords[filtered_indices, 1],0]

        # Calculate the normal vectors of the planes
        self.normalVectors = np.zeros((self.triangulationFaces.shape[0], 3, 3))

        for i in range(self.triangulationFaces.shape[0]):
            pt_1 = self.triangulationPoints[self.triangulationFaces[i][0]]
            pt_2 = self.triangulationPoints[self.triangulationFaces[i][1]]
            pt_3 = self.triangulationPoints[self.triangulationFaces[i][2]]

            v = pt_2 - pt_1
            u = pt_3 - pt_1

            orth_base = gram_schmidt([v, u])
            self.normalVectors[i, 0] = orth_base[0] / np.linalg.norm(orth_base[0])
            self.normalVectors[i, 1] = orth_base[1] / np.linalg.norm(orth_base[1])
            self.normalVectors[i, 2] = np.cross(orth_base[0], orth_base[1]) / np.linalg.norm(
                np.cross(orth_base[0], orth_base[1]))

            # Calc rotation matrix to get a positive x direction
            # Check if the x-coordinate of normal1 is negative
            if self.normalVectors[i, 0][0] < 0:
                # Calculate the rotation matrix to rotate all vectors by 180 degrees around the third vector
                angle = np.pi  # 180 degrees in radians
                axis = self.normalVectors[i, 2] / np.linalg.norm(self.normalVectors[i, 2])
                c = np.cos(angle)
                s = np.sin(angle)
                R = np.array([[c + (1 - c) * axis[0] ** 2, (1 - c) * axis[0] * axis[1] - s * axis[2],
                               (1 - c) * axis[0] * axis[2] + s * axis[1]],
                              [(1 - c) * axis[0] * axis[1] + s * axis[2], c + (1 - c) * axis[1] ** 2,
                               (1 - c) * axis[1] * axis[2] - s * axis[0]],
                              [(1 - c) * axis[0] * axis[2] - s * axis[1], (1 - c) * axis[1] * axis[2] + s * axis[0],
                               c + (1 - c) * axis[2] ** 2]])

                # Rotate all vectors by 180 degrees around the third vector
                self.normalVectors[i, 0] = np.dot(R, self.normalVectors[i, 0])
                self.normalVectors[i, 1] = np.dot(R, self.normalVectors[i, 1])
                self.normalVectors[i, 2] = np.dot(R, self.normalVectors[i, 2])

        if plot_look_up_matrix:
            look_up_matrix = np.array(self.LookUpMatrix, dtype=np.uint8)

            # Display the LookUpMatrix
            cv2.imshow('LookUpMatrix', look_up_matrix)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def worldToPixel(self, worldPoints):
        """
        Convert world points to pixel points.
        :param worldPoints: World points
        :return: Pixel points
        """

        if len(worldPoints.shape) == 1:
            worldPoints = worldPoints.reshape(1, -1)

        assert worldPoints.shape[1] == 3, "World points must have 3 columns"

        worldPoints = worldPoints.reshape(-1, 1, 3)

        pixelPoints, _ = cv2.projectPoints(worldPoints, self.cameraRotVec, self.cameraTransVec, self.cameraMat,
                                           self.distortionCoefficients, None, self.cameraMat)

        if pixelPoints.shape[1] == 1:
            pixelPoints = pixelPoints.reshape(-1, 2)

        if pixelPoints.shape[0] == 1:
            pixelPoints = pixelPoints.reshape(2)

        return pixelPoints

    def pixelToStreePlane(self, pixelPoints, plane_point = None):
        """
        Convert pixel points to street plane points.
        :param pixelPoints: Pixel points
        :return: Street plane points
        """

        if len(pixelPoints.shape) == 1:
            pixelPoints = pixelPoints.reshape(1, -1)

        if pixelPoints.dtype != np.float64:
            pixelPoints = pixelPoints.astype(np.float64)

        assert pixelPoints.shape[1] == 2, "Pixel points must have 2 columns"

        if self.calibration_type == "Delaunay triangulation":
            streetPlanePoints = self._delaunayPixelToStreetPlane(pixelPoints, ref_plane_point = plane_point)
        elif self.calibration_type == "Homography":
            undistortedPixelPoints = cv2.undistortPoints(pixelPoints, self.cameraMat, self.distortionCoefficients, None,
                                                         self.cameraMat)
            undistortedPixelPoints = undistortedPixelPoints.reshape(
                undistortedPixelPoints.shape[0] * undistortedPixelPoints.shape[1], 2)

            numEl = undistortedPixelPoints.shape[0]
            streetPlanePoints = np.zeros((numEl, 3))
            for i in range(numEl):
                streetPlanePoints[i] = self._getStreetPlanePoint_homography(undistortedPixelPoints[i, :])
        else:
            raise NotImplementedError("Calibration type {} is not supported".format(self.calibration_type))

        return streetPlanePoints

    def _getStreetPlanePoint_homography(self, pixelPoint):

        point_image = np.array([[[pixelPoint[0], pixelPoint[1]]]], dtype=np.float32)
        point_bev_single_homography = cv2.perspectiveTransform(point_image, self._homography)
        point_bev_single_homography = np.array(point_bev_single_homography, dtype=np.float32)[0, 0, :]
        height = 0

        point_bev_single_homography = np.asarray([point_bev_single_homography[0], point_bev_single_homography[1], height])
        return point_bev_single_homography

    def _delaunayPixelToStreetPlane(self, pixelPoints, ref_plane_point=None):

        undistortedPixelPoints = cv2.undistortPoints(pixelPoints, self.cameraMat, self.distortionCoefficients, None,
                                                     self.cameraMat)
        undistortedPixelPoints = undistortedPixelPoints.reshape(
            undistortedPixelPoints.shape[0] * undistortedPixelPoints.shape[1], 2)

        numEl = undistortedPixelPoints.shape[0]

        matStreetPoints = np.zeros((numEl, 3))

        if ref_plane_point is not None:
            if ref_plane_point[0] < 0 or ref_plane_point[0] >= self.imageSize[0] or \
                    ref_plane_point[1] < 0 or ref_plane_point[1] >= self.imageSize[1]:
                triangNum = -1
            else:
                triangNum = self.LookUpMatrix[int(ref_plane_point[1]), int(ref_plane_point[0]), 0]


        for i in range(numEl):

            # Check if the point is in the image
            if ref_plane_point is None:
                if pixelPoints[i][0] < 0 or pixelPoints[i][0] >= self.imageSize[0] or \
                        pixelPoints[i][1] < 0 or pixelPoints[i][1] >= self.imageSize[1]:
                    triangNum = -1
                else:
                    triangNum = self.LookUpMatrix[int(pixelPoints[i][1]), int(pixelPoints[i][0]), 0]

            plot_look_up_matrix = False
            if plot_look_up_matrix:

                look_up_matrix = np.array(self.LookUpMatrix, dtype=np.uint8)

                # Display the LookUpMatrix
                cv2.imshow('LookUpMatrix', look_up_matrix)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                plot_look_up_matrix = False

            if triangNum == -1:
                matStreetPoints[i] = np.array([-1, -1, -1])
                continue
            else:
                plane_point_1 = self.triangulationPoints[self.triangulationFaces[triangNum][0]]

                plane_point = plane_point_1

                matStreetPoints[i] = self._getStreetPlanePoint(triangNum, plane_point, undistortedPixelPoints[i, :])

        return matStreetPoints

    def createBottomMap(self):

        # Create a bottom map
        bottom_map = np.zeros((self.imageSize[0], self.imageSize[1], 3), dtype=np.float32)

        # Iterate over all pixels
        for i in range(self.imageSize[0]):
            for j in range(self.imageSize[1]):
                # Convert pixel to street plane
                streetPlanePoint = self.pixelToStreePlane(np.array([i, j]))

                if streetPlanePoint[0,0] == -1:
                    bottom_map[i, j] = [-1, -1, -1]
                else:
                    bottom_map[i, j] = (streetPlanePoint[0, :] - self.cameraPosition)

        self.bottom_map = bottom_map

        return bottom_map

    def createBottomDepthMap(self):

        # Create a bottom map
        bottom_depth_map = np.zeros((self.imageSize[0], self.imageSize[1], 1), dtype=np.float32)

        # Iterate over all pixels
        for i in range(self.imageSize[0]):
            for j in range(self.imageSize[1]):

                streetPlanePoint = self.pixelToStreePlane(np.array([i, j]))

                if np.amin(streetPlanePoint) <= 0:
                    bottom_depth_map[i, j] = -1
                    continue

                back_pixel = self.worldToPixel(streetPlanePoint)
                if np.linalg.norm(back_pixel - np.array([i, j])) > 2:
                    bottom_depth_map[i, j] = -1
                    continue

                # camera Point
                cameraPoint = self.worldToCam(streetPlanePoint)
                depth = cameraPoint[0, 2]

                if depth < 0:
                    bottom_depth_map[i, j] = -1
                    continue

                # Use pixel and depth to calc world point
                pixel_point = np.array([i, j]).reshape(1, 2)
                cam_coord = self.cameraMatInv.dot(np.array([pixel_point[0, 0], pixel_point[0, 1], 1])) * depth
                test_world = self.camToWorld(cam_coord)

                diff_world = np.linalg.norm(test_world - streetPlanePoint)

                if diff_world > 0.5:
                    bottom_depth_map[i, j] = -1
                    continue

                test_cam = self.worldToCam(test_world)
                diff_cam = np.linalg.norm(test_cam - cam_coord)

                assert diff_cam < 0.1, "Error in cameraToWorld or worldToCamera function"

                bottom_depth_map[i, j] = depth

        # num pixels with depth not -1
        num_pixels = np.sum(bottom_depth_map != -1)
        num_pixels_all = bottom_depth_map.size
        if self._verbose:
            print("Percentage of pixels with depth: ", num_pixels / num_pixels_all)

        return bottom_depth_map

    def createCamToWorldProjection(self):
        return self.cameraRTInv

    def createWorldToCamProjection(self):
        return self.cameraRT

    def getCameraIntrinsics(self):
        return self.cameraMat

    def getCameraIntrinsicsInv(self):
        return self.cameraMatInv

    def getPronjectionMatrix(self):
        return self.projectionMatrix


    def worldToCam(self, worldPoints):
        """
        Convert world points to camera points.
        :param worldPoints: World points
        :return: Camera points
        """

        if len(worldPoints.shape) == 1:
            worldPoints = worldPoints.reshape(1, -1)

        assert worldPoints.shape[1] == 3, "World points must have 3 columns"

        # Create homogeneous coordinates
        worldPointsHomogeneous = np.ones((worldPoints.shape[0], 4))
        worldPointsHomogeneous[:, :3] = worldPoints

        # Calculate camera points
        camPoints = np.matmul(worldPointsHomogeneous, np.transpose(self.cameraRT))
        camPoints = camPoints / camPoints[:, 3].reshape(-1, 1)


        return camPoints[:, :3]

    def calc_camera_frustum(self, horizontal_fov=31, vertical_fov=24, detection_distance=100):
        """
        Calculate the camera frustum in world coordinates.

        Parameters:
        - x_c, y_c, z_c: Camera position in world coordinates.
        - rotMat: Rotation matrix (3x3) of the camera.
        - horizontal_fov: Horizontal field of view in degrees.
        - vertical_fov: Vertical field of view in degrees.
        - detection_distance: Distance to the far plane.

        Returns:
        - cameraDetEdges: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max.
        - far_plane_corners_world: 4 corners of the far plane in world coordinates.
        """

        # Convert FOV to radians
        horizontal_fov_rad = np.radians(horizontal_fov)
        vertical_fov_rad = np.radians(vertical_fov)

        # Calculate the width and height of the far plane
        far_width = 2 * detection_distance * np.tan(horizontal_fov_rad / 2)
        far_height = 2 * detection_distance * np.tan(vertical_fov_rad / 2)

        # Define corners of the far plane in camera coordinates
        far_plane_corners = np.array([
            [-far_width / 2, -far_height / 2, detection_distance],  # Bottom-left
            [far_width / 2, -far_height / 2, detection_distance],  # Bottom-right
            [far_width / 2, far_height / 2, detection_distance],  # Top-right
            [-far_width / 2, far_height / 2, detection_distance]  # Top-left
        ])

        # Transform far plane corners to world coordinates
        far_plane_corners_world = self.camToWorld(far_plane_corners)

        # Calculate bounding box of the frustum
        x_min = np.min(far_plane_corners_world[:, 0])
        x_max = np.max(far_plane_corners_world[:, 0])
        y_min = np.min(far_plane_corners_world[:, 1])
        y_max = np.max(far_plane_corners_world[:, 1])
        z_min = np.min(far_plane_corners_world[:, 2])
        z_max = np.max(far_plane_corners_world[:, 2])

        cameraDetEdges = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min,
                          "z_max": z_max}

        return cameraDetEdges, far_plane_corners_world

    def bevToImage(self, bevPoints):

        # Create homogenous coordinates
        bevPointsHomogeneous = np.ones((bevPoints.shape[0], 4))
        bevPointsHomogeneous[:, :3] = bevPoints

        # Calculate image points
        imagePoints = np.matmul(bevPointsHomogeneous, np.transpose(self.projectionMatrix))

        # Normalize image points
        imagePoints = imagePoints / imagePoints[:, 2].reshape(-1, 1)

        imagePoints_rounded = np.round(imagePoints[:,:2]).astype(int)

        valid_mask = (
                (imagePoints_rounded[:, 0] >= 0) &
                (imagePoints_rounded[:, 0] < 640) &
                (imagePoints_rounded[:, 1] >= 0) &
                (imagePoints_rounded[:, 1] < 480)
        )

        imagePoints_rounded = imagePoints_rounded[valid_mask]

        # Distort image points
        imagePoints_rounded= cv2.undistortPoints(imagePoints_rounded, self.cameraMat, self.distortionCoefficients, None, self.cameraMat)


        return imagePoints[:, :2]


    def camToWorld(self, camPoints):
        """
        Convert camera points to world points.
        :param camPoints: Camera points
        :return: World points
        """

        if len(camPoints.shape) == 1:
            camPoints = camPoints.reshape(1, -1)

        assert camPoints.shape[1] == 3, "Camera points must have 3 columns"

        # Create homogeneous coordinates
        camPointsHomogeneous = np.ones((camPoints.shape[0], 4))
        camPointsHomogeneous[:, :3] = camPoints

        # Calculate world points
        worldPoints = np.matmul(self.cameraRTInv, np.transpose(camPointsHomogeneous))
        worldPoints = worldPoints / worldPoints[3, :]

        return worldPoints.T[:, :3]


    def _getStreetPlanePoint(self, triangNum, plane_point, pixelPoints):
        """
        Get the street plane point.
        :param pt_1: Point 1
        :param pt_2: Point 2
        :param pt_3: Point 3
        :param pixelPoint: Pixel point
        :return: Street plane point
        """

        if self.calibration_type == "Delaunay triangulation":
            # Get the normal vector of the plane
            plane_normal = self.normalVectors[triangNum, 2]

            # Get the intersection point
            intersection_point = self.projectPixelPointToPlane(plane_point, plane_normal, pixelPoints)

            return intersection_point

        else:
            raise NotImplementedError("Calibration type {} is not supported".format(self.calibration_type))


    def projectPixelPointToPlane(self, plane_point, plane_normals, pixel_point):

        if len(plane_normals.shape) == 2:
            plane_normal = plane_normals[0, :]
        else:
            plane_normal = plane_normals
        plane_normal = allign_normal_vector_for_hnf(plane_normal, plane_point)

        # Calculate vector from camera position to plane point
        vector_cam_to_plane = plane_point - self.cameraPosition
        dot_product_plane_cam = np.dot(plane_normal, vector_cam_to_plane) # distance from camera to plane

        # Prepare homogeneous coordinates of undistorted pixel point
        if len(pixel_point.shape) == 1:
            pixel_point = pixel_point.reshape(1, -1)
        undistortedPixels_homogeneous = np.ones((pixel_point.shape[0], 3))
        undistortedPixels_homogeneous[:, :2] = pixel_point.reshape(-1, 2)

        # Calculate dot product of plane normal with transformed undistorted pixel points
        dot_product_plane_transformed = np.matmul(undistortedPixels_homogeneous,
                                                  np.transpose(self.projectionMatrix03Inv))
        plane_normal = np.repeat(plane_normal.reshape(1, 3), len(undistortedPixels_homogeneous), axis=0)
        dot_product_plane_transformed = np.dot(plane_normal, np.transpose(dot_product_plane_transformed))

        # Calculate scaling factor for each pixel
        dot_product_plane_cam = np.repeat(dot_product_plane_cam, len(undistortedPixels_homogeneous))
        scaling_factors = dot_product_plane_cam / dot_product_plane_transformed

        # Calculate world points
        worldPoints = np.matmul(scaling_factors * undistortedPixels_homogeneous -
                                    np.repeat(self.projectionMatrix[:, 3].reshape(1, 3),
                                              len(undistortedPixels_homogeneous), axis=0),
                                    np.transpose(self.projectionMatrix03Inv))

        return worldPoints

    def getCameraPosition(self):
        return self.cameraPosition


    def get_bottom_map(self):
        if self.bottom_map is not None:
            return self.bottom_map
        else:
            return self.createBottomMap()

    def get_z_coordinate_from_x_y(self, x, y):
        """
        Get the z coordinate from the x and y coordinates.
        :return: Z coordinate
        """
        if self.create_bottom_map:
            x_help = x- self.cameraPosition[0]
            y_help = y - self.cameraPosition[1]
            if self.bottom_map is None:
                raise ValueError("Bottom map is not created. Set create_bottom_map to True in the constructor.")

            # Find the closest point in the bottom map
            idx = np.argmin(np.linalg.norm(self.bottom_map[:, :, :2] - np.array([x_help, y_help]), axis=2))
            z_value = self.bottom_map[idx // self.bottom_map.shape[1], idx % self.bottom_map.shape[1], 2]
            z_value = z_value + self.cameraPosition[2]
            if z_value == -1:
                return None
            else:
                return z_value

        else:
            raise NotImplementedError("Z coordinate from x and y is not implemented yet")




