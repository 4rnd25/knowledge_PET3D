"""
Created on Aug 07 12:27

@author: ISAC - pettirsch
"""
import cv2
import yaml
import numpy as np
import argparse
import tkinter as tk
from tkinter import messagebox


from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader


def click_event(event, x, y, flags, param):
    global coordinates
    global image

    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        figure.append((x, y))
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        if len(figure) > 1:
            cv2.line(image, figure[-2], figure[-1], (0, 0, 255), 2)
        cv2.imshow('Zone_and_Line_definition', image)


def ask_save_confirmation():
    """
    Displays a confirmation dialog asking if the user wants to save and update the config.
    Returns:
        - 'replace': if the user wants to replace existing zones.
        - 'append': if the user wants to add to existing zones.
        - None: if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    result = messagebox.askyesnocancel(
        "Save Configuration",
        "Do you want to save the polygons and update the configuration?\n\n"
        "Click 'Yes' to replace existing zones, 'No' to add to existing zones, or 'Cancel' to abort."
    )
    root.destroy()  # Destroy the root window

    if result is True:  # User clicked 'Yes'
        return 'replace'
    elif result is False:  # User clicked 'No'
        return 'append'
    else:  # User clicked 'Cancel'
        return None


def write_polygon_to_config(config_path, polygons, action):
    """
    Update the config file to include polygon definitions in GeoJSON-compatible format.
    Parameters:
        config_path (str): Path to the config file.
        polygons (list): List of new polygons.
        action (str): 'replace' to replace existing zones, 'append' to add to existing zones.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure the `detection_zone_config` exists
    if 'detection_zone_config' not in config['videoprocessing_config']:
        config['videoprocessing_config']['detection_zone_config'] = {
            'enabled': True,
            'zones': []
        }

    # Convert polygons to GeoJSON-style coordinates
    for poly_idx, polygon in enumerate(polygons):
        vertices = [[int(point[0]), int(point[1])] for point in polygon]

        if action == 'replace' and poly_idx == 0:
            # Replace existing zones with new polygons
            config['videoprocessing_config']['detection_zone_config']['zones'] = [
                {
                    'id': poly_idx+1,
                    'name': f"Zone_{poly_idx+1}",
                    'vertices': vertices
                }
            ]
        elif action == 'append' or poly_idx > 0:
            # Append new polygons to the existing zones
            next_id = max((zone.get('id', 0) for zone in config['videoprocessing_config']['detection_zone_config']['zones']), default=0) + 1
            config['videoprocessing_config']['detection_zone_config']['zones'].append(
                {
                    'id': next_id,
                    'name': f"Zone_{next_id}",
                    'vertices': vertices
                }
            )

    # Write the updated config back to the file
    with open(config_path, 'w') as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,  # Ensures block style for the file
            sort_keys=False           # Keeps the original order of keys in the config
        )

    print(f"Polygons written to config file at {config_path} with action: {action}")



def main(config, config_path, plot_current=False, verbose=False):
    global figure
    global image
    global coordinates

    image = None
    figure = []
    coordinates = []

    # Create video reader
    video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                               config['database_config'])

    # Get the first image
    while True:
        image = video_reader.get_next_frame()
        if image is not None:
            break

    # Draw a rectangle with a 5-pixel distance to the image border
    height, width = image.shape[:2]  # Get the dimensions of the image
    #cv2.rectangle(image, (5, 5), (width - 5, height - 5), (255, 255, 255),
    #              thickness=1)  # Green rectangle with 1-pixel thickness

    # Create a window and set the callback function
    cv2.namedWindow('Zone_and_Line_definition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Zone_and_Line_definition', 1280, 720)
    cv2.setMouseCallback('Zone_and_Line_definition', click_event)

    polygons = []

    while True:
        # Display the image and wait for a key press
        cv2.imshow('Zone_and_Line_definition', image)

        key = cv2.waitKey(1) & 0xFF

        # Plot current polygons from config
        if plot_current:
            color = (39,171,87)
            for zone in config['videoprocessing_config']['detection_zone_config']['zones']:
                vertices = zone['vertices']
                for i in range(len(vertices)):
                    cv2.line(image, tuple(vertices[i]), tuple(vertices[(i + 1) % len(vertices)]), color, 4)
            # Save image
            filename = config['input_config']['filename'] + "_detection_zone.png"
            cv2.imwrite(filename, image)




        if key == ord('n'):  # Create new polygon
            if len(figure) < 3:
                print("A polygon requires at least 3 points.")
            else:
                # Close the polygon if the first and last points are near
                if np.linalg.norm(np.array(figure[0]) - np.array(figure[-1])) < 20:
                    figure[-1] = figure[0]

                polygons.append(figure)
                print(f"Polygon created with {len(figure)} vertices.")
                figure = []

        elif key == ord('q'):  # Quit and ask to save
            if len(polygons) > 0:
                action = ask_save_confirmation()
                if action == 'replace' or action == 'append':
                    write_polygon_to_config(config_path, polygons, action)
                else:
                    print("Polygons were not saved.")
            else:
                print("No polygons to save.")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/10_20001_Aachen_KrefelderStr_PassStr_2.yaml',
                        help='Path to the config file')
    parser.add_argument('--plot_current', type=bool, default=True, help='Plot current zone')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    main(config, args.config,args.plot_current, args.verbose)
