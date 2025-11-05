"""
Created on May 07 2025 07:46

@author: ISAC - pettirsch
"""

import os
import cv2
import sys


class ConflictVideoPlayer:
    def __init__(self, superoutputFolder=None, conflictVideoCreator = None, verbose=False):
        self.superoutputFolder = superoutputFolder
        self.conflictVideoCreator = conflictVideoCreator
        self.verbose = verbose

    def show_conflict(self, recordName, conflict_indicator, idVehicle1, idVehicle2, class1, class2, cluster1, cluster2,
                      recordID, timeStamp, timeStamp_micro, value, maneuver):


        recordFolder = os.path.join(self.superoutputFolder, recordName)
        conflictFolder = os.path.join(recordFolder, "Conflicts", conflict_indicator + "_3s", "Conflict_Videos")

        # Iterate through all videos in conflict folder if pattern lower(class1)_idVehicle1_lower(class2)_idVehicle2 in name select video
        search_string = f"{class1.lower()}_{idVehicle1}_{cluster1}_{class2.lower()}_{idVehicle2}_{cluster2}"
        video_files = [f for f in os.listdir(conflictFolder) if
                       search_string in f and f.endswith('.mp4')]

        if len(video_files) == 0:
            search_string = f"{class2.lower()}_{idVehicle2}_{cluster2}_{class1.lower()}_{idVehicle1}_{cluster1}"
            video_files = [f for f in os.listdir(conflictFolder) if
                           search_string in f and f.endswith('.mp4')]


        if len(video_files) == 0:
            if self.conflictVideoCreator is None:
                print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
                return None, None, None
            else:
                print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
                print("Creating video...")
                self.conflictVideoCreator.create_conflict_video(recordName, recordID, conflict_indicator, idVehicle1,
                                                                idVehicle2, class1, class2, cluster1, cluster2,
                                                                conflictFolder, timeStamp, timeStamp_micro,
                                                                maneuver, value)
                # Retry finding the video
                search_string = f"{class1.lower()}_{idVehicle1}_{cluster1}_{class2.lower()}_{idVehicle2}_{cluster2}"
                video_files = [f for f in os.listdir(conflictFolder) if
                               search_string in f and f.endswith('.mp4')]
                if len(video_files) == 0:
                    print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
                    return None, None, None

        video_file = video_files[0]

        # Repeatly show video until 'q' is pressed
        video_path = os.path.join(conflictFolder, video_file)
        print(f"Playing video: {video_path}")
        # determine a starting delay (ms) from the video’s FPS
        cap = cv2.VideoCapture(video_path)
        playback_fps = 60
        delay = int(1000.0 / playback_fps)
        state = 'conflict' # states: 'conflict', 'maneuver', 'rule', 'done'
        conflict_flag = False
        maneuver = None
        rule_flag = False
        maneuvers = [
            'Following', 'Turn_left_across_path', 'Turn_right_across_path', 'Crossing',
            'Head-on', 'Merging', 'Diverging'
        ]

        cont = True
        skip = False
        while cont:
            # rewind to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break



                # Overlay instructions
                instructions = []
                if state == 'conflict':
                    instructions = ["Press 'c' to Confirm conflict", "Press 'x' to reject conflict"]
                elif state == 'maneuver':
                    instructions = [f"Press '{i}' for {m}" for i, m in enumerate(maneuvers, start=1)]
                elif state == 'rule':
                    instructions = ["Press 'y' to confirm rule", "Press 'n' to reject rule"]

                # Draw instructions on the frame
                for i, instruction in enumerate(instructions):
                    cv2.putText(frame, instruction, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                cv2.imshow(f"{video_file}", frame)
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('s'):  # slower
                    playback_fps = 30
                    delay = int(1000.0 / playback_fps)
                    print(f"[Slower] new delay = {delay} ms")
                elif key == ord('f'):  # faster
                    playback_fps = 120
                    delay = int(1000.0 / playback_fps)
                    print(f"[Faster] new delay = {delay} ms")
                elif key == ord('q'):  # quit altogether
                    print("[Quit] bye!")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord('a'): #Skip
                    print("[Skip] skipping to next video")
                    cont = False
                    skip = True
                    break

                if state == 'conflict':
                    if key == ord('c'):
                        conflict_flag = True
                        print("[Confirm] conflict confirmed")
                        state = 'maneuver'
                    elif key == ord('x'):
                        conflict_flag = False
                        print("[Reject] conflict rejected")
                        state = 'maneuver'
                elif state == 'maneuver':
                    # ASCII '0' is 48, so '1' is 49, '7' is 48 + len(maneuvers)
                    low = ord('1')
                    high = ord('0') + len(maneuvers)

                    if low <= key <= high:
                        idx = key - ord('1')  # 49 → 0, 50 → 1, …
                        maneuver = maneuvers[idx]
                        print(f"Maneuver selected: {maneuver}")
                        state = 'rule'
                elif state == 'rule':
                    if key == ord('y'):
                        rule_flag = True
                        print("[Confirm] rule confirmed")
                        state = 'done'
                    elif key == ord('n'):
                        rule_flag = False
                        print("[Reject] rule rejected")
                        state = 'done'

                if state == 'done':
                    print("[Done] moving to next video")
                    cont = False
                    break

        # clean up windows when done
        cap.release()
        cv2.destroyAllWindows()

        if not skip:
            return conflict_flag, maneuver, rule_flag
        else:
            return None, None, None

    def check_conflict(self, recordName, conflict_indicator, idVehicle1, idVehicle2, class1, class2, cluster1, cluster2):

        recordFolder = os.path.join(self.superoutputFolder, recordName)
        conflictFolder = os.path.join(recordFolder, "Conflicts", conflict_indicator + "_3s", "Conflict_Videos")

        # Iterate through all videos in conflict folder if pattern lower(class1)_idVehicle1_lower(class2)_idVehicle2 in name select video
        search_string = f"{class1.lower()}_{idVehicle1}_{cluster1}_{class2.lower()}_{idVehicle2}_{cluster2}"
        video_files = [f for f in os.listdir(conflictFolder) if
                       search_string in f and f.endswith('.mp4')]

        if len(video_files) == 0:
            search_string = f"{class2.lower()}_{idVehicle2}_{cluster2}_{class1.lower()}_{idVehicle1}_{cluster1}"
            video_files = [f for f in os.listdir(conflictFolder) if
                           search_string in f and f.endswith('.mp4')]


        if len(video_files) == 0:
            print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
            return None

        video_file = video_files[0]

        # Repeatly show video until 'q' is pressed
        video_path = os.path.join(conflictFolder, video_file)
        print(f"Playing video: {video_path}")
        # determine a starting delay (ms) from the video’s FPS
        cap = cv2.VideoCapture(video_path)
        playback_fps = 120
        delay = int(1000.0 / playback_fps)
        state = 'conflict' # states: 'conflict', 'maneuver', 'rule', 'done'
        conflict_flag = False

        cont = True
        skip = False
        while cont:
            # rewind to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Overlay instructions
                instructions = []
                if state == 'conflict':
                    instructions = ["Press 'c' to Confirm conflict", "Press 'x' to reject conflict", "Press 'r' if wrong maneuver"]

                # Draw instructions on the frame
                for i, instruction in enumerate(instructions):
                    cv2.putText(frame, instruction, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                cv2.imshow(f"{video_file}", frame)
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('s'):  # slower
                    playback_fps = 30
                    delay = int(1000.0 / playback_fps)
                    print(f"[Slower] new delay = {delay} ms")
                elif key == ord('f'):  # faster
                    playback_fps = 180
                    delay = int(1000.0 / playback_fps)
                    print(f"[Faster] new delay = {delay} ms")
                elif key == ord('q'):  # quit altogether
                    print("[Quit] bye!")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord('a'): #Skip
                    print("[Skip] skipping to next video")
                    cont = False
                    skip = True
                    break

                if state == 'conflict':
                    if key == ord('c'):
                        conflict_flag = 1
                        print("[Confirm] conflict confirmed")
                        state = 'done'
                    elif key == ord('x'):
                        conflict_flag = 0
                        print("[Reject] conflict rejected")
                        state = 'done'
                    elif key == ord('r'):
                        conflict_flag = 2
                        print("[Recheck] conflict invalid")
                        state = 'done'

                if state == 'done':
                    print("[Done] moving to next video")
                    cont = False
                    break

        # clean up windows when done
        cap.release()
        cv2.destroyAllWindows()

        if not skip:
            return conflict_flag
        else:
            return None

    def complete_check(self, recordName, recordID, conflict_indicator, idVehicle1, idVehicle2, class1, class2, cluster1, cluster2,
                       maneuver, rule_flag, conflict_flag, timeStamp, timeStamp_micro, value):

        recordFolder = os.path.join(self.superoutputFolder, recordName)
        conflictFolder = os.path.join(recordFolder, "Conflicts", conflict_indicator + "_3s", "Conflict_Videos")

        # Iterate through all videos in conflict folder if pattern lower(class1)_idVehicle1_lower(class2)_idVehicle2 in name select video
        search_string = f"{class1.lower()}_{idVehicle1}_{cluster1}_{class2.lower()}_{idVehicle2}_{cluster2}"
        video_files = [f for f in os.listdir(conflictFolder) if
                       search_string in f and f.endswith('.mp4')]

        if len(video_files) == 0:
            search_string = f"{class2.lower()}_{idVehicle2}_{cluster2}_{class1.lower()}_{idVehicle1}_{cluster1}"
            video_files = [f for f in os.listdir(conflictFolder) if
                           search_string in f and f.endswith('.mp4')]


        if len(video_files) == 0:
            if self.conflictVideoCreator is None:
                print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
                return None, None, None, None, None
            else:
                print(f"No video found for conflict {conflict_indicator} between vehicles {idVehicle1} and {idVehicle2}.")
                print("Creating video...")
                self.conflictVideoCreator.create_conflict_video(recordName, recordID, conflict_indicator, idVehicle1,
                                                                idVehicle2, class1, class2, cluster1, cluster2,
                                                                conflictFolder, timeStamp, timeStamp_micro,
                                                                maneuver, value)
                # Retry finding the video
                search_string = f"{class1.lower()}_{idVehicle1}_{cluster1}_{class2.lower()}_{idVehicle2}_{cluster2}"
                video_files = [f for f in os.listdir(conflictFolder) if
                               search_string in f and f.endswith('.mp4')]
                if len(video_files) == 0:
                    print(f"No video found after creating video for conflict {conflict_indicator} between vehicles "
                          f"{idVehicle1} and {idVehicle2}.")
                    return None, None, None, None, None

        video_file = video_files[0]

        # Repeatly show video until 'q' is pressed
        video_path = os.path.join(conflictFolder, video_file)
        print(f"Playing video: {video_path}")
        # determine a starting delay (ms) from the video’s FPS
        cap = cv2.VideoCapture(video_path)
        playback_fps = 120
        delay = int(1000.0 / playback_fps)
        state = 'conflict' # states: 'conflict', 'maneuver', 'rule', 'cluster', 'maneuver_correction', 'cluster_correction1', 'cluster_correction2', 'done'
        all_maneuvers = [
            'Following', 'Turn_left_across_path', 'Turn_right_across_path', 'Crossing',
            'Head-on', 'Merging', 'Diverging'
        ]

        cont = True
        skip = False
        cluster1_str = ""
        cluster2_str = ""
        while cont:
            # rewind to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Overlay instructions
                instructions = []
                if state == 'conflict':
                    instructions = ["Current conflict {}".format(conflict_flag), "'c' confirm 'x' reject"]
                elif state == 'maneuver':
                    instructions = ["Current maneuver: {}".format(maneuver),
                                    "'c' confirm 'x' reject"]
                elif state == 'rule':
                    instructions = ["Current rule: {}".format(rule_flag),
                                    "'c' confirm 'x' reject"]
                elif state == 'cluster':
                    instructions = ["Current Cluster 1: {} Cluster 2: {}".format(cluster1,cluster2),
                                    "'c' confirm 'x' reject"]
                elif state == 'maneuver_correction':
                    instructions = [f"Press '{i}' for {m}" for i, m in enumerate(all_maneuvers, start=1)]
                elif state == 'cluster_correction1':
                    instruction = ["Enter new cluster for vehicle 1 (current: {})".format(cluster1)]
                elif state == 'cluster_correction2':
                    instruction = ["Enter new cluster for vehicle 2 (current: {})".format(cluster2)]

                # Draw instructions on the frame
                for i, instruction in enumerate(instructions):
                    cv2.putText(frame, instruction, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                cv2.imshow(f"{video_file}", frame)
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('s'):  # slower
                    playback_fps = 30
                    delay = int(1000.0 / playback_fps)
                    print(f"[Slower] new delay = {delay} ms")
                elif key == ord('f'):  # faster
                    playback_fps = 180
                    delay = int(1000.0 / playback_fps)
                    print(f"[Faster] new delay = {delay} ms")
                elif key == ord('q'):  # quit altogether
                    print("[Quit] bye!")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord('a'): #Skip
                    print("[Skip] skipping to next video")
                    cont = False
                    skip = True
                    break

                if state == 'conflict':
                    if key == ord('c'):
                        print("[Confirm] conflict confirmed")
                        state = 'maneuver'
                    elif key == ord('x'):
                        if conflict_flag:
                            conflict_flag = False
                        else:
                            conflict_flag = True
                        print("[Reject] conflict rejected")
                        state = 'maneuver'
                elif state == 'maneuver':
                    if key == ord('c'):
                        print("[Confirm] maneuver confirmed")
                        state = 'rule'
                    elif key == ord('x'):
                        print("[Reject] maneuver rejected")
                        state = 'maneuver_correction'
                elif state == 'rule':
                    if key == ord('c'):
                        print("[Confirm] rule confirmed")
                        state = 'cluster'
                    elif key == ord('x'):
                        if rule_flag:
                            rule_flag = False
                        else:
                            rule_flag = True
                        print("[Reject] rule rejected")
                        state = 'cluster'
                elif state == 'cluster':
                    if key == ord('c'):
                        print("[Confirm] cluster confirmed")
                        state = 'done'
                    elif key == ord('x'):
                        print("[Reject] cluster rejected")
                        state = 'cluster_correction1'
                elif state == 'maneuver_correction':
                    # ASCII '0' is 48, so '1' is 49, '7' is 48 + len(maneuvers)
                    low = ord('1')
                    high = ord('0') + len(all_maneuvers)

                    if low <= key <= high:
                        idx = key - ord('1')  # 49 → 0, 50 → 1, …
                        maneuver = all_maneuvers[idx]
                        print(f"Maneuver selected: {maneuver}")
                        state = 'cluster'
                elif state == 'cluster_correction1':
                    # digit?
                    if ord('0') <= key <= ord('9'):
                        # allow up to two digits (so max "50")
                        if len(cluster1_str) < 2:
                            cluster1_str += chr(key)
                    elif key in (8, 127):  # backspace on most systems
                        cluster1_str = cluster1_str[:-1]
                    elif key == 13:  # Enter
                        try:
                            val = int(cluster1_str)
                            if 0 <= val <= 50:
                                cluster1 = val
                                cluster1_str = ''  # reset buffer
                                state = 'cluster_correction2'
                                print(f"[Cluster1] set to {cluster1}")
                            else:
                                print("[Error] must be 0–50; try again")
                                cluster1_str = ''
                        except ValueError:
                            print("[Error] no valid number entered; try again")
                            cluster1_str = ''

                elif state == 'cluster_correction2':
                    if ord('0') <= key <= ord('9'):
                        if len(cluster2_str) < 2:
                            cluster2_str += chr(key)
                    elif key in (8, 127):
                        cluster2_str = cluster2_str[:-1]
                    elif key == 13:
                        try:
                            val = int(cluster2_str)
                            if 0 <= val <= 50:
                                cluster2 = val
                                cluster2_str = ''
                                state = 'cluster'  # go back to the normal cluster-review
                                print(f"[Cluster2] set to {cluster2}")
                            else:
                                print("[Error] must be 0–50; try again")
                                cluster2_str = ''
                        except ValueError:
                            print("[Error] no valid number entered; try again")
                            cluster2_str = ''

            if state == 'done':
                    print("[Done] moving to next video")
                    cont = False
                    break

        # clean up windows when done
        cap.release()
        cv2.destroyAllWindows()

        if not skip:
            return conflict_flag, maneuver, rule_flag, cluster1, cluster2
        else:
            return None, None, None, None, None