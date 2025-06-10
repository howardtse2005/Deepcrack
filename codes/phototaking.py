import pyrealsense2 as rs
import numpy as np
import cv2
import os
import threading
import time

def main():
    # Create output directories if they don't exist
    output_dir_color = "data/Crack/test/rgb"
    output_dir_depth = "data/Crack/test/depth"
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_depth, exist_ok=True)
    
    # Initialize counters
    place_counter = 1      # i value (increments with 'b')
    position_counter = 1   # j value (increments with 'a')
    
    # Flag to control the main loop
    running = True
    
    # Event flags for keyboard actions
    take_picture_a = threading.Event()
    take_picture_b = threading.Event()
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    print("Press 'a' to take a picture (increment position)")
    print("Press 'b' to take a picture (increment place)")
    print("Press 'q' to quit")
    print("Enter commands in this terminal window")
    
    # Function to handle keyboard input from terminal
    def keyboard_input():
        nonlocal running
        while running:
            try:
                key = input().strip().lower()
                if key == 'a':
                    take_picture_a.set()
                elif key == 'b':
                    take_picture_b.set()
                elif key == 'q':
                    print("Quitting...")
                    running = False
            except Exception as e:
                print(f"Input error: {e}")
    
    # Start keyboard input thread
    input_thread = threading.Thread(target=keyboard_input)
    input_thread.daemon = True
    input_thread.start()
    
    try:
        while running:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            # Align depth to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames (renamed for clarity)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not aligned_color_frame:
                continue
            
            # Convert images to numpy arrays 
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())
            
            # Apply colormap to depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            # Check for window close (x button)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
            # Handle 'a' key (take picture, increment position)
            if take_picture_a.is_set():
                print(f"Key 'a' pressed - Taking picture at place: {place_counter}, position: {position_counter}")
                
                # Generate filenames
                color_filename = f"{place_counter}_{position_counter}.jpg"
                depth_filename = f"{place_counter}_{position_counter}_depth.jpg"
                
                # Save images
                cv2.imwrite(os.path.join(output_dir_color, color_filename), color_image)
                
                # Convert depth to 8-bit for saving as jpg
                normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(os.path.join(output_dir_depth, depth_filename), normalized_depth)
                
                print(f"Saved {color_filename} to {output_dir_color}")
                print(f"Saved {depth_filename} to {output_dir_depth}")
                
                # Increment position counter and make sure changes are visible
                position_counter += 1
                take_picture_a.clear()
                
            # Handle 'b' key (increment place, then take picture, then increment position)
            if take_picture_b.is_set():
                # First increment place counter and reset position counter
                place_counter += 1
                position_counter = 1
                print(f"Key 'b' pressed - Moved to place: {place_counter}, reset position to: {position_counter}")
                
                # Then generate filenames with updated counters
                color_filename = f"{place_counter}_{position_counter}.jpg"
                depth_filename = f"{place_counter}_{position_counter}_depth.jpg"
                
                # Save images
                cv2.imwrite(os.path.join(output_dir_color, color_filename), color_image)
                
                # Convert depth to 8-bit for saving as jpg
                normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(os.path.join(output_dir_depth, depth_filename), normalized_depth)
                
                print(f"Saved {color_filename} to {output_dir_color}")
                print(f"Saved {depth_filename} to {output_dir_depth}")
                
                # Increment position counter for the next picture
                position_counter += 1
                
                take_picture_b.clear()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
                
    finally:
        # Stop the loop in the input thread
        running = False
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()