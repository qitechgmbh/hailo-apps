#!/usr/bin/env python3
import os
import sys
import queue
import threading
from functools import partial
from types import SimpleNamespace
from pathlib import Path
import collections
import numpy as np
from flask import Flask, Response
import cv2

# Initialize Flask app for streaming
app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()

import socket
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def send_detections_to_socket(detections_list):
    # Path to the socket file created by the receiver
    socket_path = "/tmp/qitech_object_detections.sock"
    json_data = json.dumps(detections_list, cls=NumpyEncoder).encode('utf-8')
    # 1. Check if the socket file actually exists before trying to connect
    if not os.path.exists(socket_path):
        return # Or log a warning
    try:
        # 2. Create a Unix Domain Socket (AF_UNIX)
        # SOCK_STREAM is used for TCP-like reliable connection
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # 3. Connect to the receiver
        client.connect(socket_path)
        client.sendall(json_data)
    except Exception as e:
        print(f"Socket Error: {e}")
    finally:
        # 6. Always close the connection
        client.close()

def generate_frames():
    """Generator function for the MJPEG stream."""
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


# -----------------------------------------------------------------------------
# Ensure repository root is available in sys.path
# -----------------------------------------------------------------------------
repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break

if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.toolbox import (
    InputContext,
    VisualizationSettings,
    init_input_source,
    get_labels,
    load_json_file,
    preprocess,
    visualize,
    FrameRateTracker,
    stop_after_timeout
)
from hailo_apps.python.core.common.defines import (
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS,
)
from hailo_apps.python.core.common.parser import get_standalone_parser
from hailo_apps.python.core.common.hailo_logger import (
    get_logger,
    init_logging,
    level_from_args,
)
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import (
    inference_result_handler,
)
from hailo_apps.python.core.common.core import handle_and_resolve_args

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)

def parse_args():
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Run object detection with optional tracking and performance measurement."

    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Enable object tracking for detections. "
            "When enabled, detected objects will be tracked across frames using a tracking algorithm "
            "(e.g., Bytetrack). This assigns consistent IDs to objects over time, enabling temporal analysis, "
            "trajectory visualization, and multi-frame association. Useful for video processing applications."
        ),
    )

    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        ),
    )

    parser.add_argument(
        "--draw-trail",
        action="store_true",
        help=(
            "[Tracking only] Draw motion trails of tracked objects.\n"
            "Uses the last 30 positions from the tracker history."
        )
    )

    args = parser.parse_args()
    return args


def run_inference_pipeline(
    net,
    labels,
    input_context: InputContext,
    visualization_settings: VisualizationSettings,
    enable_tracking: bool = False,
    show_fps: bool = True,
    draw_trail: bool = False,
    time_to_run: int | None = None,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.
    """
    labels = get_labels(labels)
    app_dir = Path(__file__).resolve().parent
    config_path = app_dir / "config.json"
    config_data = load_json_file(str(config_path))

    stop_event = threading.Event()
    fps_tracker = FrameRateTracker() if show_fps else None
    tracker = None

    if enable_tracking:
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    post_process_callback_fn = partial(
        inference_result_handler,
        labels=labels,
        config_data=config_data,
        tracker=tracker,
        draw_trail=draw_trail,
    )

    hailo_inference = HailoInfer(net, input_context.batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(
            input_context,
            input_queue,
            width,
            height,
            None,  # Use default preprocess from toolbox
            stop_event,
        ),
        name="preprocess-thread",
    )

    infer_thread = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue, stop_event),
        name="infer-thread",
    )

    preprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    if time_to_run is not None:
        timer_thread = threading.Thread(
            target=stop_after_timeout,
            args=(stop_event, time_to_run),
            name="timer-thread",
            daemon=True,
        )
        timer_thread.start()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Stream available at http://<your-device-ip>:5000/video_feed")
    try:
        # Instead of 'visualize', we create a custom loop to process the output_queue
        while not stop_event.is_set():
            try:
                # Get processed data from the queue (blocks for a short timeout)
                batch_data = output_queue.get(timeout=1)
                if batch_data is None: break

                frame, result = batch_data

                # Apply your post-processing/drawing (formerly done inside visualize)
                # post_process_callback_fn returns the frame with boxes drawn
                processed_frame, detections = post_process_callback_fn(frame, result)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                send_detections_to_socket(detections)
                print(detections)
                # Update the global frame for the Flask streamer
                global output_frame
                with frame_lock:
                    output_frame = processed_frame

            except queue.Empty:
                continue
    finally:
        stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    if show_fps:
        logger.info(fps_tracker.frame_rate_summary())
    logger.success("Processing completed successfully.")
    if visualization_settings.save_stream_output or input_context.has_images:
        logger.info(f"Saved outputs to '{visualization_settings.output_dir}'.")


def infer(hailo_inference, input_queue, output_queue, stop_event):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    # Limit number of concurrent async inferences
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        if stop_event.is_set():
            continue  # Skip processing if stop signal is set

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )


        while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
            pending_jobs.popleft().wait(10000)

        # Run async inference
        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    # Release resources and context
    hailo_inference.close()
    output_queue.put(None)


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))

def main() -> None:
    args = parse_args()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME)

    input_context = InputContext(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        video_unpaced=args.video_unpaced,
    )

    input_context = init_input_source(input_context)

    visualization_settings = VisualizationSettings(
        output_dir=args.output_dir,
        save_stream_output=args.save_output,
        output_resolution=args.output_resolution,
        no_display=args.no_display,
    )

    run_inference_pipeline(
        net=args.hef_path,
        labels=args.labels,
        input_context=input_context,
        visualization_settings=visualization_settings,
        enable_tracking=args.track,
        show_fps=args.show_fps,
        draw_trail=args.draw_trail,
        time_to_run=args.time_to_run
    )

if __name__ == "__main__":
    main()