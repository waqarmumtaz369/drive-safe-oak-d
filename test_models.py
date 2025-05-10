# Main script for driver distraction detection using DepthAI and OAK-D CM4
# This script is designed for use on a Raspberry Pi OS (Raspbian) with an OAK-D CM4 device.
# It uses DepthAI to run neural network models for detecting people, cell phones, and seatbelt usage in video frames.
# The script processes a video file, draws bounding boxes and labels, logs results, and can output an annotated video.

import depthai as dai  # Import DepthAI Python API for device and pipeline control
import cv2             # OpenCV for video I/O and image processing
import numpy as np     # NumPy for numerical operations
import os              # OS utilities for file and directory management
import time            # Time utilities for measuring processing time
import argparse        # For parsing command-line arguments
import datetime        # For timestamping logs

# Main function to process a video for person, phone, and seatbelt detection
# This function sets up the pipeline, loads models, processes each frame, and logs results

def process_video(video_path, output_path=None, show_video=True, seatbelt_model_path="blob/seatbelt.blob", 
               batch_size=1, frame_interval=1, optimize_for_rvc2=True, inference_threads=1):
    """
    Process a video file to detect people and cell phones using DepthAI
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path to save processed video (optional, removed)
        show_video (bool): Whether to display the video during processing
        seatbelt_model_path (str): Path to seatbelt classifier model
        batch_size (int): Number of frames to process at once (1-4)
        frame_interval (int): Process every Nth frame (1=all frames)
        optimize_for_rvc2 (bool): Apply RVC2-specific optimizations
        inference_threads (int): Number of inference threads (1-2, must match pool frames)
    """
    # Define colors for drawing bounding boxes and labels
    GREEN = (0, 255, 0)    # For person with seatbelt
    RED = (0, 0, 255)      # For person without seatbelt (OpenCV uses BGR)
    YELLOW = (0, 255, 255) # For phone detection
    WHITE = (255, 255, 255)
    
    # Ensure the models directory exists (required for seatbelt model)
    models_dir = os.path.dirname(os.path.abspath(seatbelt_model_path))
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading detection models...")
    # Load YOLOv5/YOLOv8 model for person and cell phone detection
    model_path = 'blob/yolov8n_coco_416x416_openvino_2022.1_6shave.blob'
    classes = {
        0: "person",
        67: "cell phone"
    }
    
    # Check if seatbelt model exists and enable seatbelt detection if available
    use_seatbelt_detection = False
    if os.path.exists(seatbelt_model_path):
        use_seatbelt_detection = True
        print(f"Using seatbelt model: {seatbelt_model_path}")
    else:
        print(f"Seatbelt model not found at {seatbelt_model_path}, seatbelt detection disabled")
    
    # Labels for seatbelt classification
    seatbelt_labels = {
        0: "Not Worn",
        1: "Worn"
    }
    
    # Create DepthAI pipeline
    pipeline = dai.Pipeline()
    
    # Set OpenVINO version for optimal RVC2 performance
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)
    
    # Define pipeline nodes for input, detection, and output
    cam_rgb = pipeline.create(dai.node.XLinkIn)
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    
    cam_rgb.setStreamName("frame")
    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("detections")
    
    # Set maximum frame size for input
    cam_rgb.setMaxDataSize(3 * 1920 * 1080)  # Maximum frame size
    
    # Configure detection network properties
    detection_nn.setBlobPath(model_path)
    detection_nn.setConfidenceThreshold(0.20)  # Lower threshold to catch more detections
    detection_nn.setNumClasses(80)
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors(np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]))
    detection_nn.setAnchorMasks({
        "side52": np.array([0, 1, 2]),
        "side26": np.array([3, 4, 5]),
        "side13": np.array([6, 7, 8])
    })
    detection_nn.setIouThreshold(0.5)
    detection_nn.setNumInferenceThreads(inference_threads)
    detection_nn.setNumNCEPerInferenceThread(2)  # Optimize NCE usage for RVC2
    detection_nn.setNumPoolFrames(inference_threads)  # Must be at least equal to number of executors
    detection_nn.input.setBlocking(False)
    
    # Link pipeline nodes
    cam_rgb.out.link(detection_nn.input)
    detection_nn.passthrough.link(xout_rgb.input)
    detection_nn.out.link(xout_nn.input)
    
    # If seatbelt detection is enabled, set up additional pipeline nodes
    if use_seatbelt_detection:
        seatbelt_in = pipeline.create(dai.node.XLinkIn)
        seatbelt_in.setStreamName("seatbelt_in")
        seatbelt_nn = pipeline.create(dai.node.NeuralNetwork)
        seatbelt_nn.setBlobPath(seatbelt_model_path)
        seatbelt_nn.setNumPoolFrames(inference_threads)
        seatbelt_nn.input.setBlocking(False)
        seatbelt_nn.input.setQueueSize(1)
        seatbelt_nn.setNumInferenceThreads(inference_threads)
        seatbelt_nn.setNumNCEPerInferenceThread(2)
        seatbelt_out = pipeline.create(dai.node.XLinkOut)
        seatbelt_out.setStreamName("seatbelt_out")
        seatbelt_in.out.link(seatbelt_nn.input)
        seatbelt_nn.out.link(seatbelt_out.input)
    
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties (width, height, fps, total frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Remove video writer logic, only show processed video
    writer = None  # No video writer

    # Connect to DepthAI device and start the pipeline
    with dai.Device(pipeline) as device:
        # Create output queues for RGB frames and detections
        q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=2, blocking=False)
        # Create input queue for sending frames
        q_in = device.getInputQueue(name="frame", maxSize=2)
        # If seatbelt detection is enabled, create input/output queues for seatbelt
        q_seatbelt_in = None
        q_seatbelt_out = None
        if use_seatbelt_detection:
            q_seatbelt_in = device.getInputQueue(name="seatbelt_in", maxSize=2)
            q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=2, blocking=False)
        frame_count = 0
        start_time = time.time()
        print(f"Processing video with {total_frames} frames...")
        # Initialize summary statistics
        total_person_detections = 0
        total_phone_detections = 0
        frames_with_detections = 0
        frames_with_seatbelt = 0
        frames_without_seatbelt = 0
        # Main loop: process each frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Skip frames for performance if frame_interval > 1
            if frame_interval > 1 and frame_count % frame_interval != 0:
                continue
            current_time = time.time() - start_time
            # Print progress every 10 frames
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                remaining_time = remaining_frames / fps_processing if fps_processing > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%) - ETA: {remaining_time:.1f}s, FPS: {fps_processing:.2f}")
            # Resize frame to 416x416 for YOLO model
            resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)
            # Convert frame to DepthAI ImgFrame and send to device
            img = dai.ImgFrame()
            img.setData(resized_frame.transpose(2, 0, 1).flatten())
            img.setType(dai.RawImgFrame.Type.BGR888p)
            img.setWidth(resized_frame.shape[1])
            img.setHeight(resized_frame.shape[0])
            img.setTimestamp(time.monotonic())
            q_in.send(img)
            # Get detection results from device
            in_nn = q_nn.tryGet()
            # Initialize detection counters for this frame
            person_count = 0
            phone_count = 0
            seatbelt_status = "Not checked"
            # Helper function to calculate bounding box area
            def get_bbox_area(detection):
                width = detection.xmax - detection.xmin
                height = detection.ymax - detection.ymin
                return width * height
            if in_nn is not None:
                detections = in_nn.detections
                if len(detections) > 0:
                    frames_with_detections += 1
                # Separate detections by class
                person_detections = []
                phone_detections = []
                # Process each detection
                for i, detection in enumerate(detections):
                    class_id = detection.label
                    # Classify detection
                    if class_id == 0:  # person
                        person_detections.append(detection)
                        person_count += 1
                        total_person_detections += 1
                    elif class_id == 67:  # cell phone
                        phone_detections.append(detection)
                        phone_count += 1
                        total_phone_detections += 1
                # If person detected and seatbelt detection enabled, run seatbelt classifier on largest person
                if person_detections and use_seatbelt_detection:
                    largest_person = max(person_detections, key=get_bbox_area)
                    x1 = largest_person.xmin * width
                    x2 = largest_person.xmax * width
                    y1 = largest_person.ymin * height
                    y2 = largest_person.ymax * height
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    # Ensure bounding box is within frame
                    box_left = max(0, box_left)
                    box_top = max(0, box_top)
                    box_width = min(width - box_left, box_width)
                    box_height = min(height - box_top, box_height)
                    # Extract ROI for seatbelt detection
                    if box_width > 0 and box_height > 0:
                        person_roi = frame[box_top:box_top+box_height, box_left:box_left+box_width]
                        try:
                            # Resize ROI for seatbelt model, keep aspect ratio
                            person_h, person_w, _ = person_roi.shape
                            target_size = 224
                            if person_h > person_w:
                                new_h = target_size
                                new_w = int(person_w * (target_size / person_h))
                            else:
                                new_w = target_size
                                new_h = int(person_h * (target_size / person_w))
                            resized_roi = cv2.resize(person_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            # Pad to 224x224
                            delta_w = target_size - new_w
                            delta_h = target_size - new_h
                            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                            left, right = delta_w // 2, delta_w - (delta_w // 2)
                            seatbelt_input = cv2.copyMakeBorder(resized_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            # Create ImgFrame for seatbelt classifier
                            seatbelt_img = dai.ImgFrame()
                            seatbelt_img.setData(seatbelt_input.transpose(2, 0, 1).flatten())
                            seatbelt_img.setType(dai.RawImgFrame.Type.BGR888p)
                            seatbelt_img.setWidth(224)
                            seatbelt_img.setHeight(224)
                            seatbelt_img.setTimestamp(time.monotonic())
                            # Send to seatbelt classifier
                            q_seatbelt_in.send(seatbelt_img)
                            # Get seatbelt classification result
                            seatbelt_result = q_seatbelt_out.tryGet()
                            if seatbelt_result is not None:
                                seatbelt_data = np.array(seatbelt_result.getFirstLayerFp16())
                                print(f"  Seatbelt classifier output: {seatbelt_data}")
                                seatbelt_class = np.argmax(seatbelt_data)
                                confidence = seatbelt_data[seatbelt_class]
                                if seatbelt_class == 0 and confidence < 0.8:
                                    seatbelt_status = "Uncertain"
                                else:
                                    seatbelt_status = seatbelt_labels[seatbelt_class]
                                if seatbelt_class == 1:
                                    frames_with_seatbelt += 1
                                else:
                                    frames_without_seatbelt += 1
                                seatbelt_color = GREEN if seatbelt_class == 1 else RED
                                cv2.putText(frame, f"Seatbelt {seatbelt_status}", 
                                          (box_left + 5, box_top + 50), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.85, seatbelt_color, 3)
                        except Exception as e:
                            seatbelt_status = "Error"
                # Draw bounding box for largest person
                if person_detections:
                    largest_person = max(person_detections, key=get_bbox_area)
                    x1 = largest_person.xmin * width
                    x2 = largest_person.xmax * width
                    y1 = largest_person.ymin * height
                    y2 = largest_person.ymax * height
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    person_color = GREEN if seatbelt_status == "Worn" else RED
                    cv2.rectangle(frame, (box_left, box_top), 
                                (box_left + box_width, box_top + box_height), person_color, 3)
                    label = f"person: {largest_person.confidence:.2f}"
                    cv2.putText(frame, label, (box_left + 5, box_top + 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.85, person_color, 3)
                # Draw bounding box for largest phone
                if phone_detections:
                    largest_phone = max(phone_detections, key=get_bbox_area)
                    x1 = largest_phone.xmin * width
                    x2 = largest_phone.xmax * width
                    y1 = largest_phone.ymin * height
                    y2 = largest_phone.ymax * height
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    cv2.rectangle(frame, (box_left, box_top), 
                                (box_left + box_width, box_top + box_height), YELLOW, 3)
                    label = f"Phone Detected: {largest_phone.confidence:.2f}"
                    cv2.putText(frame, label, (box_left, box_top - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 3)
            # Only display the frame in a window
            if show_video or True:  # Always show video
                try:
                    cv2.imshow('Driver Monitoring', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    if "Can't initialize GTK backend" in str(e):
                        print("Warning: Display not available, disabling video preview")
                        break
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        processing_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        print(f"Average FPS: {frame_count/processing_time:.2f}")
        # Print summary to console
        print("\nDetection Summary:")
        print(f"  Frames with detections: {frames_with_detections}/{frame_count} ({100*frames_with_detections/frame_count:.1f}%)")
        print(f"  Total person detections: {total_person_detections}")
        print(f"  Total phone detections: {total_phone_detections}")
        if use_seatbelt_detection:
            print(f"  Frames with seatbelt: {frames_with_seatbelt}")
            print(f"  Frames without seatbelt: {frames_without_seatbelt}")

# Entry point for command-line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Distraction Detection using DepthAI")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--show", action="store_true", help="Show video preview (if display available)")
    parser.add_argument("--seatbelt", help="Path to seatbelt classifier model", default="blob/seatbelt.blob")
    parser.add_argument("--batch", type=int, help="Batch size for processing (1-4)", default=1)
    parser.add_argument("--interval", type=int, help="Process every Nth frame", default=1)
    parser.add_argument("--optimize", action="store_true", help="Apply RVC2-specific optimizations", default=True)
    parser.add_argument("--threads", type=int, help="Number of inference threads (1-2)", default=1)
    args = parser.parse_args()
    if args.video:
        process_video(args.video, show_video=args.show, seatbelt_model_path=args.seatbelt,
                    batch_size=args.batch, frame_interval=args.interval, optimize_for_rvc2=args.optimize,
                    inference_threads=args.threads)
    else:
        print("Please provide a video path with --video argument")