# Vehicle-Distance-Detection-on-Single-Camera


This script is designed to perform object detection using the YOLOv5 model. It processes images or videos, detects objects, and annotates them with bounding boxes, labels, and confidence scores. The script also estimates distances between objects using their bounding box dimensions.

---

## Code Description

### 1. **Import Libraries**:
   - **`argparse`**: Handles command-line arguments for model configuratio  n and input/output settings.
   - **`torch`** and **`cudnn`**: Utilize PyTorch for model operations and optimize GPU computations.
   - Utility modules like `utils.datasets`, `utils.utils`, and `utils.downloads` manage data handling and preprocessing.

---

### 2. **Command-Line Arguments**:
   Define parameters for detection, including:
   - `--weights`: Path to the YOLOv5 model weights.
   - `--source`: Input source (image, video, webcam, or directory).
   - `--output`: Folder to save detection results.
   - `--img-size`: Size of the input image for inference.
   - `--conf-thres`: Confidence threshold for object detection.
   - `--iou-thres`: IoU threshold for Non-Max Suppression (NMS).
   - `--view-img`: Display the results.
   - `--save-txt`: Save detection results to a text file.
   - `--device`: Specify CUDA device or use CPU.
   - Other parameters for additional functionality like augmentations or filtering classes.

---

### 3. **Model Initialization**:
   - Loads the YOLOv5 model with specified weights.
   - Configures the model for GPU or CPU usage.
   - Converts model to half precision (`FP16`) for faster inference if CUDA is available.

---

### 4. **Data Handling**:
   - **Webcam/Video Input**: Uses `LoadStreams` for real-time processing.
   - **Static Images**: Uses `LoadImages` for batch processing of image files.
   - Dynamically adjusts input size based on model stride.

---

### 5. **Inference and Detection**:
   - **Run Inference**: Processes input data through the model to get predictions.
   - **Non-Max Suppression (NMS)**: Filters overlapping bounding boxes using confidence and IoU thresholds.
   - **Distance Calculation**: Estimates the distance of objects based on bounding box diagonal length and a pre-defined scaling factor.

---

### 6. **Annotations**:
   - Draws bounding boxes and labels on detected objects.
   - Annotates each object with class name, confidence, and estimated distance.
   - Results can be saved as images or videos.

---

### 7. **Results Saving**:
   - Saves annotated frames to the specified output folder.
   - Writes detection details to `.txt` files if enabled.
   - Displays results in a window if `--view-img` is set.

---

## How to Use This Code


### 1. Install Required Packages: 
   Ensure you have PyTorch and required libraries installed.
   ```bash
   pip install torch torchvision opencv-python numpy
   ```

### 2. Run the Script

Use the command below, replacing parameters as needed:

   ```bash
python detect.py --weights weights/yolov5s.pt --source data/images --output results --img-size 640
   ```

## Notes

- Model Compatibility: Ensure the weights file matches the YOLOv5 version used.
- GPU Usage: Use --device 0 for GPU acceleration, or --device cpu to run on the CPU.
- Distance Estimation: Adjust the scaling factor for accurate distance measurements based on camera and scene calibration.
- Performance Optimization: Enable cudnn.benchmark for improved GPU performance with fixed-size inputs.