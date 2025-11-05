"""
Created on Apr 07 2024 10:53

@author: ISAC - pettirsch
"""

import cv2
import numpy as np

def plot_3d_box(frame, corners_3d_pix, bottom_front_left_idx, bottom_front_right_idx,
                bottom_back_right_idx, bottom_back_left_idx, top_front_left_idx, top_front_right_idx,
                top_back_right_idx, top_back_left_idx, box_2d=None,
                color=(0, 255, 0), thickness=2, mark_corners=True):
    # Draw bottom face
    cv2.line(frame, tuple(corners_3d_pix[bottom_front_left_idx].astype(int)),
             tuple(corners_3d_pix[bottom_front_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_front_right_idx].astype(int)),
             tuple(corners_3d_pix[bottom_back_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_back_right_idx].astype(int)),
             tuple(corners_3d_pix[bottom_back_left_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_back_left_idx].astype(int)),
             tuple(corners_3d_pix[bottom_front_left_idx].astype(int)), color, thickness)

    # Draw top face
    cv2.line(frame, tuple(corners_3d_pix[top_front_left_idx].astype(int)),
             tuple(corners_3d_pix[top_front_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[top_front_right_idx].astype(int)),
             tuple(corners_3d_pix[top_back_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[top_back_right_idx].astype(int)),
             tuple(corners_3d_pix[top_back_left_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[top_back_left_idx].astype(int)),
             tuple(corners_3d_pix[top_front_left_idx].astype(int)), color, thickness)

    # Draw lines connecting top and bottom face
    cv2.line(frame, tuple(corners_3d_pix[bottom_front_left_idx].astype(int)),
             tuple(corners_3d_pix[top_front_left_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_front_right_idx].astype(int)),
             tuple(corners_3d_pix[top_front_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_back_right_idx].astype(int)),
             tuple(corners_3d_pix[top_back_right_idx].astype(int)), color, thickness)
    cv2.line(frame, tuple(corners_3d_pix[bottom_back_left_idx].astype(int)),
             tuple(corners_3d_pix[top_back_left_idx].astype(int)), color, thickness)

    # Mark each corner with different color
    if mark_corners:
        frame = cv2.circle(frame, tuple(corners_3d_pix[bottom_front_left_idx].astype(int)), 5, (255, 0, 0), -1)  # blue
        frame = cv2.circle(frame, tuple(corners_3d_pix[bottom_front_right_idx].astype(int)), 5, (0, 255, 0),
                           -1)  # green
        frame = cv2.circle(frame, tuple(corners_3d_pix[bottom_back_right_idx].astype(int)), 5, (0, 0, 255), -1)  # red
        frame = cv2.circle(frame, tuple(corners_3d_pix[bottom_back_left_idx].astype(int)), 5, (255, 255, 0),
                           -1)  # light blue
        frame = cv2.circle(frame, tuple(corners_3d_pix[top_front_left_idx].astype(int)), 5, (0, 255, 255), -1)  # yellow
        frame = cv2.circle(frame, tuple(corners_3d_pix[top_front_right_idx].astype(int)), 5, (255, 0, 255), -1)  # pink
        frame = cv2.circle(frame, tuple(corners_3d_pix[top_back_right_idx].astype(int)), 5, (255, 255, 255),
                           -1)  # white
        frame = cv2.circle(frame, tuple(corners_3d_pix[top_back_left_idx].astype(int)), 5, (0, 0, 0), -1)  # black

    if box_2d is not None:
        cv2.rectangle(frame, (box_2d[0], box_2d[1]), (box_2d[2], box_2d[3]), (0, 0, 0), 1)

    return frame


def plot_3d_corners(corners_3d_img, img, color, line_thickness, label=None):
    # Draw bottom face
    cv2.line(img, tuple(corners_3d_img[0].astype(int)),
             tuple(corners_3d_img[1].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[1].astype(int)),
             tuple(corners_3d_img[2].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[2].astype(int)),
             tuple(corners_3d_img[3].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[3].astype(int)),
             tuple(corners_3d_img[0].astype(int)), color, line_thickness)

    # Draw top face
    cv2.line(img, tuple(corners_3d_img[4].astype(int)),
             tuple(corners_3d_img[5].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[5].astype(int)),
             tuple(corners_3d_img[6].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[6].astype(int)),
             tuple(corners_3d_img[7].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[7].astype(int)),
             tuple(corners_3d_img[4].astype(int)), color, line_thickness)

    # Draw lines connecting top and
    cv2.line(img, tuple(corners_3d_img[0].astype(int)),
             tuple(corners_3d_img[4].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[1].astype(int)),
             tuple(corners_3d_img[5].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[2].astype(int)),
             tuple(corners_3d_img[6].astype(int)), color, line_thickness)
    cv2.line(img, tuple(corners_3d_img[3].astype(int)),
             tuple(corners_3d_img[7].astype(int)), color, line_thickness)

    if label:
        label_pos_x = int(np.round(np.min(corners_3d_img[:, 0])))
        label_pos_y = int(np.round(np.min(corners_3d_img[:, 1])))
        cv2.putText(img, label, (label_pos_x, label_pos_y - 2), 0, 0.5, color, thickness=1,
                    lineType=cv2.LINE_AA)

    return img
