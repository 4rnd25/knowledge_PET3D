"""
Created on Apr 15 2025 10:15

@author: ISAC - pettirsch
"""

import argparse
import datetime
import psutil
import requests
import sys
import time
import cv2
import os
import pandas as pd

from flask import Flask, Response
from threading import Thread

from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from Videoprocessing.OutputManager.outputmanager import OutputManager
from Videoprocessing.VideoProcessor.videoprocessor import VideoProcessor
from Videoprocessing.OutputVideoWriting.outputvideowriter import OutputVideoWriter
from Videoprocessing.Saver.trajectorysaver import TrajectorySaver
from Videoprocessing.TrajectoryEnhancer.trajectory_enhancer import TrajectoryEnhancer

### Main Video processing ###
def main(config, verbose=False):
    # ---- Processing statistics ----
    stats_rows = []  # list of dicts for per-record rows
    overall_videos = 0
    overall_frames = 0
    overall_duration = 0.0
    source_fps_list = []  # to compute overall_fps (mean input fps)

    global video_stream
    global shutdown_flag
    shutdown_flag = False

    # Create video reader
    video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                               config['database_config'])

    # Create output manager
    output_manager = OutputManager(config['output_config']['output_folder'], video_reader.getFilename(),
                                   verbose=verbose)

    # Create Video Processor -> Detection and tracking
    video_processor = VideoProcessor(config["videoprocessing_config"], video_reader.get_frame_size(),
                                     config['database_config'], recordID=video_reader.get_recordID(),
                                     fps=video_reader.fps, verbose=verbose)

    # Trajectory Enhancer
    trajectory_enhancer = TrajectoryEnhancer(config["videoprocessing_config"]['trajectory_enhancer_config'],
                                             cost_threshold_dict=config["videoprocessing_config"]['matching_config']['cost_thresholds'],
                                             persTrans = video_processor.perspectiveTransform,
                                             verbose=verbose)


    # Create output writer
    output_video_writer = OutputVideoWriter(video_reader.getFilename(), output_manager.get_output_path(),
                                            config['output_video_config'], video_reader.fps,
                                            video_reader.get_frame_size(), verbose=verbose)

    # Create Trajectory saver
    trajectory_saver = TrajectorySaver(config['trajectory_saver_config'], config['database_config'],
                                       video_reader.get_recordID(),
                                       outputFolder=output_manager.get_output_path(),
                                       filename = video_reader.getFilename(),
                                       enhanced = True,
                                       verbose=False)

    # Start processing
    start_time = video_reader.get_start_time()
    frame_id = 0
    frame_100_time = time.time()

    rec_frames = 0
    rec_t0 = None
    current_rec_id = video_reader.get_recordID()

    import datetime

    def round_timestamp_to_frame_step(start_time, frame_id, fps):
        # Step in microseconds (1 / 30 seconds)
        step_us = round(1e6 / fps)  # e.g., 1e6 / 30 = 33333.333... → 33333
        total_us = round(frame_id * step_us)
        return start_time + datetime.timedelta(microseconds=total_us)

    new_record = True
    video_already_processed = False
    while True:
        process_time = time.time()

        # Get frame
        frame = video_reader.get_next_frame()

        if frame is None and video_reader.is_done():
            break

        if new_record:
            video_already_processed = trajectory_saver.check_processed()
            if video_already_processed:
                frame = None
            new_record = False
            current_rec_id = video_reader.get_recordID()
            rec_frames = 0
            rec_t0 = time.perf_counter()
            source_fps_list.append(float(video_reader.fps))
            new_record = False

        if (frame is None or video_reader.should_restart(start_time)) and not video_reader.is_done():
            if not video_already_processed and rec_t0 is not None:

                # Finish all tracks
                active_tracks, finished_tracks = video_processor.stop_all_tracks()

                # Start enhancing
                trajectory_enhancer.add_tracks(finished_tracks)
                trajectory_enhancer.enhance_trajectories()

                # Save trajectories
                trajectory_saver.save_enhanced_tracks(trajectory_enhancer.get_enhanced_trajectories())

                rec_t1 = time.perf_counter()
                rec_dur = rec_t1 - rec_t0
                rec_fps = (rec_frames / rec_dur) if rec_dur > 0 else 0.0

                stats_rows.append({
                    "rec_id": int(current_rec_id) if current_rec_id is not None else -1,
                    "num_frames": int(rec_frames),
                    "processing_duration_in_sec": round(rec_dur, 6),
                    "fps": round(rec_fps, 3),
                })

                overall_videos += 1
                overall_frames += rec_frames
                overall_duration += rec_dur
            else:
                print("Video already processed, skipping...")

            video_reader.restart()

            output_manager.restart(video_reader.getFilename())
            trajectory_saver.restart(recordID=video_reader.get_recordID(),
                                     outputFolder=output_manager.get_output_path(),
                                     filename=video_reader.getFilename())
            video_already_processed = trajectory_saver.check_processed()
            if video_already_processed:
                new_record = True
                frame = None
                continue
            rec_frames = 0
            rec_t0 = time.perf_counter()
            current_rec_id = video_reader.get_recordID()
            source_fps_list.append(float(video_reader.fps))
            trajectory_enhancer.restart()
            video_processor.restart(config["videoprocessing_config"], video_reader.get_frame_size(),
                                    config['database_config'], recordID=video_reader.get_recordID())
            output_video_writer.restart(output_manager.get_output_path())
            trajectory_enhancer.restart()
            trajectory_saver.restart(recordID=video_reader.get_recordID(),outputFolder=output_manager.get_output_path(),
                                       filename = video_reader.getFilename())
            new_record = True

            frame_id = 0
            start_time = video_reader.get_start_time()
            continue
        elif frame is None and video_reader.is_done():
            break

        curr_time_stamp_video = round_timestamp_to_frame_step(start_time, frame_id, video_reader.fps)
        curr_time_stamp_system = datetime.datetime.now()

        # Process frame
        active_tracks, finished_tracks = video_processor.process_frame(frame, frame_id, curr_time_stamp_video,
                                                                          curr_time_stamp_system)

        # Add tracks to trajectory enhancer
        trajectory_enhancer.add_tracks(finished_tracks)

        # Get processed frame
        processed_frame = video_processor.get_processed_frame()
        rec_frames += 1

        # Write output video
        output_video_writer.write_frame(processed_frame, frame_id)

        # Update the global video stream for browser
        video_stream = processed_frame

        # Print summary (every 100 frames): FPS, memory usage, detected objects, conflicts
        if frame_id % 100 == 0:
            print("-------------------------------------------------")
            print(f"Frame ID: {frame_id}")
            print(f"Curr FPS: {100 / (time.time() - frame_100_time)}")
            print(f"Memory usage: {psutil.virtual_memory().percent}")
            # print(f"Detected objects: {results['object_detection']['num_objects']}")
            # print(f"Conflicts: {results['conflict']['num_conflicts']}")
            print("-------------------------------------------------")
            frame_100_time = time.time()
        frame_id += 1

    # Finish all tracks
    active_tracks, finished_tracks = video_processor.stop_all_tracks()

    # Start enhancing
    trajectory_enhancer.add_tracks(finished_tracks)
    trajectory_enhancer.enhance_trajectories()

    # Save trajectories
    trajectory_saver.save_enhanced_tracks(trajectory_enhancer.get_enhanced_trajectories())

    # Finalize last record if not yet saved
    if rec_t0 is not None and rec_frames > 0:
        rec_t1 = time.perf_counter()
        rec_dur = rec_t1 - rec_t0
        rec_fps = (rec_frames / rec_dur) if rec_dur > 0 else 0.0

        stats_rows.append({
            "rec_id": int(current_rec_id) if current_rec_id is not None else -1,
            "num_frames": int(rec_frames),
            "processing_duration_in_sec": round(rec_dur, 6),
            "fps": round(rec_fps, 3),
        })

        overall_videos += 1
        overall_frames += rec_frames
        overall_duration += rec_dur

    # Release resources
    video_reader.release()
    video_processor.release()
    output_video_writer.release()

    # ---- Write summary CSV ----
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config_name = os.path.splitext(os.path.basename(config.get('__config_path__', 'config')))[0]

    start_rec = config['database_config']['startrecordID']
    end_rec = config['database_config']['endrecordID']

    summary_filename = f"{timestamp_str}_{config_name}_start_rec_{start_rec}_end_rec_{end_rec}.csv"
    out_dir = config.get('output_config', {}).get('output_folder', '.')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, summary_filename)

    # Create dataframe from per-record stats
    df_stats = pd.DataFrame(stats_rows, columns=[
        "rec_id", "num_frames", "processing_duration_in_sec", "fps"
    ])

    # --- compute numeric totals ---
    total_videos = len(df_stats)
    total_frames = df_stats["num_frames"].sum()
    total_duration = df_stats["processing_duration_in_sec"].sum()
    total_fps = (total_frames / total_duration) if total_duration > 0 else 0.0

    # Append totals as numeric row (no text labels)
    overall_row = pd.DataFrame([{
        "rec_id": total_videos,
        "num_frames": int(total_frames),
        "processing_duration_in_sec": round(total_duration, 6),
        "fps": round(total_fps, 3),
    }])

    # Concatenate and save
    df_stats = pd.concat([df_stats, overall_row], ignore_index=True)
    df_stats.to_csv(summary_path, index=False)

    print(f"Wrote summary CSV to: {summary_path}")


#### Streaming #########
app = Flask(__name__)
video_stream = None

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shutdown', methods=['POST'])
def shutdown():
    global shutdown_flag
    shutdown_flag = True
    return 'Shutting down Flask server...'


def generate_stream():
    global video_stream
    global shutdown_flag
    while not shutdown_flag:
        if video_stream is not None:
            ret, jpeg = cv2.imencode('.jpg', video_stream)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

########################

####### Main #############
if __name__ == '__main__':
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=108, help='Start record ID')
    parser.add_argument('--endRecID', type=int, default=109, help='End record ID')
    parser.add_argument('--port', type=int, default=8052, help='Port for the Flask server')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)
    config['__config_path__'] = args.config

    # Adapt config based on command line arguments
    if args.startRecID != -1:
        config['database_config']['startrecordID'] = args.startRecID
    if args.endRecID != -1:
        config['database_config']['endrecordID'] = args.endRecID
    if args.port != -1:
        config["stream_config"]["port"] = args.port

    # Initialize Flask server
    flask_thread = Thread(
        target=lambda: app.run(host=config["stream_config"]["host"], port=config["stream_config"]["port"], debug=False,
                               use_reloader=False))
    flask_thread.daemon = True  # Make the thread a daemon so it stops with the main program
    flask_thread.start()

    # Start video processing
    main(config, args.verbose)

    # Shutdown Flask server
    try:
        requests.post(f'http://{config["stream_config"]["host"]}:{config["stream_config"]["port"]}/shutdown')
    except Exception as e:
        print(f"Error shutting down Flask")
    print("Flask server stopped")

    # exit the program
    sys.exit(0)
