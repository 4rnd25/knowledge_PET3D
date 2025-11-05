"""
Created on Jan 27 2025 11:44

@author: ISAC - pettirsch
"""
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from shapely.ops import nearest_points

class DetectionZoneFilter:
    def __init__(self, detection_zone_config, perspectiveTransform=None):
        """
        Initialize the DetectionZoneFilter with a spatial index (STRtree) for detection zones.

        Args:
            detection_zone_config (dict): Configuration containing detection zones.
        """
        self.detection_zones = []  # List of Shapely Polygons
        self.spatial_index = None  # STRtree for spatial indexing
        self.perspectiveTransform = perspectiveTransform

        if detection_zone_config and detection_zone_config.get("enabled", False):
            for zone in detection_zone_config.get("zones", []):
                # Extract vertices and create a Shapely Polygon
                vertices = zone["vertices"]
                if vertices:
                    polygon = Polygon(vertices)
                    self.detection_zones.append(polygon)

            # Create the spatial index if there are detection zones
            if self.detection_zones:
                self.spatial_index = STRtree(self.detection_zones)

    def validDetection(self, obj_point):
        """
        Check if the object (point) is within any detection zone.

        Args:
            obj_point (tuple): The point to check (x, y).

        Returns:
            bool: True if the point is inside any detection zone, False otherwise.
        """
        if not self.spatial_index:
            return False

        shapely_point = Point(obj_point)

        # Query the spatial index to find candidate polygons
        candidate_zones = self.spatial_index.query(shapely_point)

        # Check if the point is inside any of the candidate polygons
        for zone in candidate_zones:
            if self.detection_zones[int(zone)].contains(shapely_point):
                return True

        return False

    def getImageBorders(self):
        """
        Get the image borders of the detection zones.

        Returns:
            list: List of image borders for the detection zones.
        """
        return [list(zone.exterior.coords) for zone in self.detection_zones]

    def check_distance_to_border(self, key_point, dim_thresh_det):
        # Check if keypoint is in Zone and if the distance to the border is greater than the threshold
        for zone in self.detection_zones:
            if zone.contains(Point(key_point)):
                # Find the closest point on the exterior (border) of the polygon
                keypoint_geom = Point(key_point)
                nearest_border_point, _ = nearest_points(zone.exterior, keypoint_geom)

                # Calculate the distance between the keypoint and the nearest border point  (distance in meters)
                nearest_border_point = np.array([nearest_border_point.x, nearest_border_point.y])
                if self.perspectiveTransform is not None:
                    nearest_border_point = self.perspectiveTransform.pixelToStreePlane(nearest_border_point)
                    key_point = self.perspectiveTransform.pixelToStreePlane(key_point)
                dist = np.linalg.norm(nearest_border_point - key_point)

                if dist < dim_thresh_det:
                   return True
        return False
