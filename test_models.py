import depthai as dai
import cv2
import numpy as np
import os
import time
import argparse
import datetime

def process_video(video_path, output_path="abc.mp4", show_video=False, seatbelt_model_path="blob/seatbelt.blob"):
    """
    Process a video file to detect people and cell phones using DepthAI
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path to save processed video (optional)
        show_video (bool): Whether to display the video during processing
        seatbelt_model_path (str): Path to seatbelt classifier model
    """
    # Define colors
    GREEN = (0, 255, 0)    # For person with seatbelt
    RED = (0, 0, 255)      # For person without seatbelt (OpenCV uses BGR)
    YELLOW = (0, 255, 255) # For phone detection
    WHITE = (255, 255, 255)
    
    # Create models directory if needed
    models_dir = os.path.dirname(os.path.abspath(seatbelt_model_path))
    os.makedirs(models_dir, exist_ok=True)
    
    # Add after the models_dir creation
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    print("Loading detection models...")
    # Load the YOLOv5 model for person and cell phone detection
    model_path = 'blob/yolov8n_coco_416x416_openvino_2022.1_6shave.blob'
    classes = {
        0: "person",
        67: "cell phone"
    }
    
    # Check if seatbelt model exists
    use_seatbelt_detection = False
    if os.path.exists(seatbelt_model_path):
        use_seatbelt_detection = True
        print(f"Using seatbelt model: {seatbelt_model_path}")
    else:
        print(f"Seatbelt model not found at {seatbelt_model_path}, seatbelt detection disabled")
    
    # Seatbelt classification labels
    seatbelt_labels = {
        0: "Not Worn",
        1: "Worn"
    }
    
    # Create log file for detailed frame-by-frame analysis
    log_file = open("detection_log.txt", "w")
    log_file.write(f"Detection log started at {datetime.datetime.now()}\n")
    log_file.write(f"Video: {video_path}\n")
    log_file.write(f"Model: {model_path}\n")
    if use_seatbelt_detection:
        log_file.write(f"Seatbelt model: {seatbelt_model_path}\n")
    log_file.write("\nFrame,Time,Detections,PersonCount,PhoneCount\n")
    log_file.flush()

    # Create DepthAI pipeline
    pipeline = dai.Pipeline()
    
    # Define sources and outputs
    cam_rgb = pipeline.create(dai.node.XLinkIn)
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    
    cam_rgb.setStreamName("frame")
    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("detections")
    
    # Properties
    cam_rgb.setMaxDataSize(3 * 1920 * 1080)  # Maximum frame size
    
    # Network specific settings
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
    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)
    
    # Linking
    cam_rgb.out.link(detection_nn.input)
    detection_nn.passthrough.link(xout_rgb.input)
    detection_nn.out.link(xout_nn.input)
    
    # Set up seatbelt detection if model exists
    if use_seatbelt_detection:
        # Create an XLinkIn node for the seatbelt input
        seatbelt_in = pipeline.create(dai.node.XLinkIn)
        seatbelt_in.setStreamName("seatbelt_in")

        # Create neural network node for seatbelt detection
        seatbelt_nn = pipeline.create(dai.node.NeuralNetwork)
        seatbelt_nn.setBlobPath(seatbelt_model_path)
        
        # Add input configuration to specify expected format
        seatbelt_nn.setNumPoolFrames(1)
        seatbelt_nn.input.setBlocking(False)
        seatbelt_nn.input.setQueueSize(1)

        # Create output node for seatbelt detection results
        seatbelt_out = pipeline.create(dai.node.XLinkOut)
        seatbelt_out.setStreamName("seatbelt_out")
        
        # Link the nodes
        seatbelt_in.out.link(seatbelt_nn.input)
        seatbelt_nn.out.link(seatbelt_out.input)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        log_file.write(f"Error: Could not open video {video_path}\n")
        log_file.close()
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log_file.write(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} total frames\n\n")
    log_file.flush()
    
    # Setup video writer if output_path is provided
    writer = None
    if output_path:
        # Check if output path ends with .mp4, if not append it
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path + '.mp4'

        # Use H.264 codec for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec instead of 'XVID'
        
        # Ensure we have valid dimensions
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if writer initialized successfully
        if not writer.isOpened():
            print(f"Error: Could not create output video at {output_path}")
            print("Trying alternative codec...")
            
            # Try alternative codec
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
            
            if not writer.isOpened():
                print("Failed to create video writer with alternative codec too.")
                writer = None
        else:
            print(f"Writing output to {output_path}")

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        # Input queue for sending frames
        q_in = device.getInputQueue(name="frame")
        
        # Seatbelt detection queues
        q_seatbelt_in = None
        q_seatbelt_out = None
        if use_seatbelt_detection:
            q_seatbelt_in = device.getInputQueue(name="seatbelt_in")
            q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=4, blocking=False)
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video with {total_frames} frames...")
        log_file.write("Beginning frame processing\n")
        
        # Summary statistics
        total_person_detections = 0
        total_phone_detections = 0
        frames_with_detections = 0
        frames_with_seatbelt = 0
        frames_without_seatbelt = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time() - start_time
            
            # Print progress
            if frame_count % 10 == 0:  # Status update every 10 frames
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                remaining_time = remaining_frames / fps_processing if fps_processing > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%) - ETA: {remaining_time:.1f}s")
            
            # Log every frame
            log_file.write(f"\n--- Frame {frame_count} (Time: {current_time:.2f}s) ---\n")
            
            resized_frame = cv2.resize(frame, (416, 416))
            # Convert frame to DepthAI format and send to device
            img = dai.ImgFrame()
            img.setData(resized_frame.transpose(2, 0, 1).flatten())
            img.setType(dai.RawImgFrame.Type.BGR888p)
            img.setWidth(resized_frame.shape[1])
            img.setHeight(resized_frame.shape[0])
            img.setTimestamp(time.monotonic())
            q_in.send(img)
            
            # Log sent frame
            log_file.write(f"Sent frame to device: size={resized_frame.shape}\n")
            
            # Get detections
            in_nn = q_nn.tryGet()
            
            # Track detections for this frame
            person_count = 0
            phone_count = 0
            seatbelt_status = "Not checked"
            
            # Function to calculate bounding box area
            def get_bbox_area(detection):
                width = detection.xmax - detection.xmin
                height = detection.ymax - detection.ymin
                return width * height
            
            if in_nn is not None:
                detections = in_nn.detections
                log_file.write(f"Received {len(detections)} detections\n")
                
                if len(detections) > 0:
                    frames_with_detections += 1
                
                # Sort detections by class
                person_detections = []
                phone_detections = []
                
                # Process each detection
                for i, detection in enumerate(detections):
                    # Get class ID
                    class_id = detection.label
                    
                    # Log all detections
                    log_file.write(f"  Detection #{i+1}: class_id={class_id}, confidence={detection.confidence:.4f}\n")
                    log_file.write(f"    Bounding box: ({detection.xmin:.4f}, {detection.ymin:.4f}) to ({detection.xmax:.4f}, {detection.ymax:.4f})\n")
                    
                    # Sort detections by class
                    if class_id == 0:  # person
                        person_detections.append(detection)
                        person_count += 1
                        total_person_detections += 1
                        log_file.write(f"    Identified as PERSON\n")
                    elif class_id == 67:  # cell phone
                        phone_detections.append(detection)
                        phone_count += 1
                        total_phone_detections += 1
                        log_file.write(f"    Identified as CELL PHONE\n")
                
                # Find largest person bounding box if any
                if person_detections and use_seatbelt_detection:
                    # Sort by area (descending)
                    largest_person = max(person_detections, key=get_bbox_area)
                    
                    # Get normalized bounding box
                    x1 = largest_person.xmin * width
                    x2 = largest_person.xmax * width
                    y1 = largest_person.ymin * height
                    y2 = largest_person.ymax * height
                    
                    # Convert to top-left format
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    
                    # Ensure box is within frame boundaries
                    box_left = max(0, box_left)
                    box_top = max(0, box_top)
                    box_width = min(width - box_left, box_width)
                    box_height = min(height - box_top, box_height)
                    
                    # Extract person ROI for seatbelt detection
                    if box_width > 0 and box_height > 0:
                        person_roi = frame[box_top:box_top+box_height, box_left:box_left+box_width].copy()
                        
                        # Perform seatbelt classification on the person ROI
                        try:
                            # Resize for the seatbelt model, preserving aspect ratio
                            person_h, person_w, _ = person_roi.shape
                            target_size = 224
                            
                            # Determine new size maintaining aspect ratio
                            if person_h > person_w:
                                new_h = target_size
                                new_w = int(person_w * (target_size / person_h))
                            else:
                                new_w = target_size
                                new_h = int(person_h * (target_size / person_w))
                            
                            resized_roi = cv2.resize(person_roi, (new_w, new_h))
                            
                            # Create a black canvas and place the resized ROI in the center
                            delta_w = target_size - new_w
                            delta_h = target_size - new_h
                            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                            left, right = delta_w // 2, delta_w - (delta_w // 2)
                            
                            seatbelt_input = cv2.copyMakeBorder(resized_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            # Create DepthAI image and send to seatbelt classifier
                            seatbelt_img = dai.ImgFrame()
                            seatbelt_img.setData(seatbelt_input.transpose(2, 0, 1).flatten())
                            seatbelt_img.setType(dai.ImgFrame.Type.BGR888p)
                            seatbelt_img.setWidth(224)
                            seatbelt_img.setHeight(224)
                            seatbelt_img.setTimestamp(time.monotonic())
                            
                            # Send to seatbelt classifier
                            q_seatbelt_in.send(seatbelt_img)
                            
                            # Get seatbelt classification result
                            seatbelt_result = q_seatbelt_out.tryGet()
                            
                            if seatbelt_result is not None:
                                # Process classification result
                                seatbelt_data = np.array(seatbelt_result.getFirstLayerFp16())
                                
                                log_file.write(f"  Seatbelt classifier output: {seatbelt_data}\n")
                                print(f"  Seatbelt classifier output: {seatbelt_data}")
                                # Binary classification output - get class with highest confidence
                                seatbelt_class = np.argmax(seatbelt_data)
                                confidence = seatbelt_data[seatbelt_class]
                                
                                if seatbelt_class == 0 and confidence < 0.8:
                                    seatbelt_status = "Uncertain"
                                else:
                                    seatbelt_status = seatbelt_labels[seatbelt_class]
                                
                                log_file.write(f"  Seatbelt classification: {seatbelt_status} (confidence: {confidence:.4f})\n")
                                
                                # Track seatbelt statistics
                                if seatbelt_class == 1:  # Wearing seatbelt
                                    frames_with_seatbelt += 1
                                else:  # Not wearing seatbelt
                                    frames_without_seatbelt += 1
                                
                                # Draw seatbelt status on frame with color based on status
                                seatbelt_color = GREEN if seatbelt_class == 1 else RED
                                cv2.putText(frame, f"Seatbelt {seatbelt_status}", 
                                          (box_left + 5, box_top + 50), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.85, seatbelt_color, 3)
                            else:
                                log_file.write("  No seatbelt classification result received\n")
                                
                        except Exception as e:
                            log_file.write(f"  Error in seatbelt detection: {str(e)}\n")
                            seatbelt_status = "Error"
                
                # Draw person bounding box if any
                if person_detections:
                    largest_person = max(person_detections, key=get_bbox_area)
                    
                    # Get normalized bounding box
                    x1 = largest_person.xmin * width
                    x2 = largest_person.xmax * width
                    y1 = largest_person.ymin * height
                    y2 = largest_person.ymax * height
                    
                    # Convert to top-left format
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    
                    # Set color based on seatbelt status
                    person_color = GREEN if seatbelt_status == "Worn" else RED
                    
                    # Draw bounding box with appropriate color
                    cv2.rectangle(frame, (box_left, box_top), 
                                (box_left + box_width, box_top + box_height), person_color, 3)
                    
                    # Draw label with same color
                    label = f"person: {largest_person.confidence:.2f}"
                    cv2.putText(frame, label, (box_left + 5, box_top + 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.85, person_color, 3)
                    
                    log_file.write(f"    Drawing largest PERSON detection (area: {get_bbox_area(largest_person):.4f})\n")
                
                # Find largest phone bounding box if any
                if phone_detections:
                    # Sort by area (descending)
                    largest_phone = max(phone_detections, key=get_bbox_area)
                    
                    # Get normalized bounding box
                    x1 = largest_phone.xmin * width
                    x2 = largest_phone.xmax * width
                    y1 = largest_phone.ymin * height
                    y2 = largest_phone.ymax * height
                    
                    # Convert to top-left format
                    box_left = int(x1)
                    box_top = int(y1)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)
                    
                    # Draw bounding box with YELLOW color
                    cv2.rectangle(frame, (box_left, box_top), 
                                (box_left + box_width, box_top + box_height), YELLOW, 3)
                    
                    # Draw label with YELLOW text
                    label = f"Phone Detected: {largest_phone.confidence:.2f}"
                    cv2.putText(frame, label, (box_left, box_top - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 3)
                    
                    
                    log_file.write(f"    Drawing largest PHONE detection (area: {get_bbox_area(largest_phone):.4f})\n")
            else:
                log_file.write("No detections received\n")
            
            # Write summary to log
            log_file.write(f"Frame summary: {person_count} people, {phone_count} phones, seatbelt: {seatbelt_status}\n")
            log_file.flush()
            
            # Write frame to output video
            if writer:
                writer.write(frame)
            
            # Display the frame
            if show_video:
                try:
                    cv2.imshow('Driver Monitoring', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        log_file.write("Processing stopped by user\n")
                        break
                except cv2.error as e:
                    # Handle error when display is not available
                    if "Can't initialize GTK backend" in str(e) and show_video:
                        print("Warning: Display not available, disabling video preview")
                        show_video = False  # Disable for future frames
        
        # Clean up
        cap.release()
        if writer:
            print(f"Finalizing video output to {output_path}")
            writer.release()
        cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        
        # Write final summary to log
        log_file.write("\n--- FINAL SUMMARY ---\n")
        log_file.write(f"Total frames processed: {frame_count}\n")
        log_file.write(f"Processing time: {processing_time:.2f} seconds\n")
        log_file.write(f"Average FPS: {frame_count/processing_time:.2f}\n")
        log_file.write(f"Frames with detections: {frames_with_detections} ({100*frames_with_detections/frame_count:.1f}%)\n")
        log_file.write(f"Total person detections: {total_person_detections}\n")
        log_file.write(f"Total phone detections: {total_phone_detections}\n")
        log_file.write(f"Average persons per frame: {total_person_detections/frame_count:.2f}\n")
        log_file.write(f"Average phones per frame: {total_phone_detections/frame_count:.2f}\n")
        
        if use_seatbelt_detection:
            log_file.write(f"Frames with seatbelt: {frames_with_seatbelt} ({100*frames_with_seatbelt/frame_count:.1f}%)\n")
            log_file.write(f"Frames without seatbelt: {frames_without_seatbelt} ({100*frames_without_seatbelt/frame_count:.1f}%)\n")
        
        # Print summary to console
        print("\nDetection Summary:")
        print(f"  Frames with detections: {frames_with_detections}/{frame_count} ({100*frames_with_detections/frame_count:.1f}%)")
        print(f"  Total person detections: {total_person_detections}")
        print(f"  Total phone detections: {total_phone_detections}")
        
        if use_seatbelt_detection:
            print(f"  Frames with seatbelt: {frames_with_seatbelt}")
            print(f"  Frames without seatbelt: {frames_without_seatbelt}")
            
        print(f"  Log file: detection_log.txt")
        
        # Close log file
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Distraction Detection using DepthAI")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--output", help="Path to output video (optional)")
    parser.add_argument("--show", action="store_true", help="Show video preview (if display available)")
    parser.add_argument("--seatbelt", help="Path to seatbelt classifier model", default="blob/seatbelt.blob")

    args = parser.parse_args()
    
    if args.video:
        process_video(args.video, args.output, show_video=False, seatbelt_model_path=args.seatbelt)
    else:
        print("Please provide a video path with --video argument")