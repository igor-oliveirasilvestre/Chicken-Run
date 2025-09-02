import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import skimage
import math
from utils import *
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import os
import glob
current_dir = os.path.dirname(os.path.abspath(__file__))



# Lista de nomes das classes para a detecção de objetos
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"
          ]


class VideoSelector:
    def __init__(self):
        self.selected_video = None
        self.selected_model = None
        
    def get_video_files(self, data_folder="data"):
        """Get all video files from the data folder"""
        if not os.path.exists(data_folder):
            print(f"Warning: {data_folder} folder not found!")
            return []
        
        # Common video extensions
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []
        
        for extension in video_extensions:
            video_files.extend(glob.glob(os.path.join(data_folder, extension)))
            video_files.extend(glob.glob(os.path.join(data_folder, extension.upper())))
        
        # Return just the filenames for display
        return [os.path.basename(video) for video in video_files]
    
    def create_gui(self):
        """Create the GUI for video and model selection"""
        root = tk.Tk()
        root.title("YOLO Video Detection - Select Video and Model")
        root.geometry("550x400")
        root.resizable(True, True)
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLO Bird Detection", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Video selection
        ttk.Label(main_frame, text="Select Video:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        video_files = self.get_video_files()
        if not video_files:
            video_files = ["No videos found in data folder"]
        
        self.video_var = tk.StringVar()
        video_dropdown = ttk.Combobox(main_frame, textvariable=self.video_var, values=video_files, 
                                     state="readonly", width=50)
        video_dropdown.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        if video_files and video_files[0] != "No videos found in data folder":
            video_dropdown.current(0)  # Select first video by default
        
        # Save data option
        self.save_data_var = tk.BooleanVar(value=False)  # Default to not saving
        save_checkbox = ttk.Checkbutton(main_frame, 
                                       text="Save detection data to runs/ folder", 
                                       variable=self.save_data_var)
        save_checkbox.grid(row=3, column=0, columnspan=2, pady=(10, 10), sticky=tk.W)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Instructions:\n• Select a video from the dropdown\n• Choose whether to save detection data\n• Click 'Start Detection' to begin bird detection",
                               font=("Arial", 10), justify=tk.LEFT)
        instructions.grid(row=4, column=0, columnspan=2, pady=(0, 30), sticky=tk.W)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
        # Start button
        start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", command=root.quit)
        cancel_button.pack(side=tk.LEFT)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        self.root = root
        
    def start_detection(self):
        """Start the detection process with selected parameters"""
        selected_video = self.video_var.get()
        
        if not selected_video or selected_video == "No videos found in data folder":
            messagebox.showerror("Error", "Please select a valid video file!")
            return
        
        self.selected_video = os.path.join("data", selected_video)
        self.selected_model = "yolov8l.pt"  # Use default model
        self.save_data = self.save_data_var.get()  # Get checkbox value
        
        # Close the GUI and start detection
        self.root.quit()
        self.root.destroy()
        
    def select_video_and_model(self):
        """Show GUI and return selected video, model, and save option"""
        self.create_gui()
        self.root.mainloop()
        return self.selected_video, self.selected_model, getattr(self, 'save_data', False)


def main(video_path, yolo_model, use_gpu=True, save_data=False):
    # Check if required files exist
    if not os.path.exists("mask.png"):
        print("Warning: mask.png not found!")
    if not os.path.exists("banner.png"):
        print("Warning: banner.png not found!")
    mask_path = os.path.join(current_dir,"mask.png")
    mask = cv2.imread(mask_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    model = YOLO(yolo_model)
    
    # Export to OpenVINO for Intel GPU acceleration
    if use_gpu:
        try:
            print("Exporting model to OpenVINO format for Intel GPU acceleration...")
            model.export(format="openvino")
            openvino_model_path = f"{yolo_model.split('.')[0]}_openvino_model"
            
            # Load OpenVINO model
            print("Loading model with Intel GPU acceleration...")
            model = YOLO(openvino_model_path, task='detect')
            
            # Check available OpenVINO devices
            import openvino as ov
            print("Available OpenVINO devices:")
            core = ov.Core()
            available_devices = core.available_devices
            print(available_devices)
            
            # For Intel GPU, we don't pass device parameter to YOLO
            # OpenVINO will handle device selection internally
            if 'GPU' in available_devices:
                print("Using Intel GPU for acceleration!")
                use_intel_gpu = True
            else:
                print("GPU not available, using CPU")
                use_intel_gpu = False
                
        except Exception as e:
            print(f"GPU acceleration setup failed: {e}")
            print("Falling back to CPU...")
            model = YOLO(yolo_model)
            use_intel_gpu = False
    else:
        print("Using CPU for inference...")
        model = YOLO(yolo_model)
        use_intel_gpu = False

    # Inicializando o objeto "tracker" usando a classe Sort com os seguintes parâmetros:
    # max_age: O número máximo de quadros (frames) que um objeto pode estar ausente antes de ser excluído
    # min_hits: O número mínimo de quadros (frames) que um objeto deve ser detectado para ser considerado rastreado
    # iou_threshold: O limiar de Intersecção sobre União (IoU) para determinar se duas bounding boxes se sobrepõem
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Lista para armazenar os IDs dos objetos contados
    counter = []

    print(f"Processing video: {video_path}")
    print("Press 'q' to quit the video processing")

    while True:
        suc, img = cap.read()
        if not suc:
            print("End of video reached or error reading frame")
            break
        
        # Resize mask to match frame dimensions if mask exists
        if mask is not None:
            frame_height, frame_width = img.shape[:2]
            mask_resized = cv2.resize(mask, (frame_width, frame_height))
            img_region = cv2.bitwise_and(img, mask_resized)
        else:
            img_region = img

        if os.path.exists("banner.png"):
            img_banner = cv2.imread("banner.png", cv2.IMREAD_UNCHANGED)
            img = cvzone.overlayPNG(img, img_banner, (0, 0))
            
        # Run inference - OpenVINO handles device selection internally for Intel GPU
        # Control whether to save detection data based on user choice
        results = model(img_region, stream=True, save_crop=save_data)

        detections = np.empty((0, 5))

        for res in results:
            boxes = res.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_label = labels[cls]

                if current_label == "bird" and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        results_from_tracker = tracker.update(detections)

        for result in results_from_tracker:
            x1, y1, x2, y2, id_res = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {"Chicken"}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if counter.count(id_res) == 0:
                counter.append(id_res)

        cv2.putText(img, str(len(counter)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        cv2.imshow("Image", img)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection completed. Total objects counted: {len(counter)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None, help='path to the video file')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='path to the YOLO model')
    parser.add_argument('--gui', action='store_true', help='use GUI for video selection')
    parser.add_argument('--cpu', action='store_true', help='force CPU inference (disable GPU acceleration)')
    parser.add_argument('--save', action='store_true', help='save detection data to runs/ folder')
    args = parser.parse_args()

    use_gpu = not args.cpu  # Use GPU unless --cpu flag is specified

    if args.gui or args.video is None:
        # Use GUI for video selection
        selector = VideoSelector()
        selected_video, selected_model, save_data = selector.select_video_and_model()
        
        if selected_video and selected_model:
            main(selected_video, selected_model, use_gpu, save_data)
        else:
            print("No video or model selected. Exiting...")
    else:
        # Use command line arguments
        main(args.video, args.model, use_gpu, args.save)