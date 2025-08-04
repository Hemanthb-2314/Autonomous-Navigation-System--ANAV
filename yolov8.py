import roboflow
from ultralytics import YOLO
import os

# --- Roboflow Credentials ---
API_KEY = "yuGCSsiwbr2YmVg9uIbs"
WORKSPACE = "mars-2iirj"
PROJECT = "mars-craters-and-safe-landing"
VERSION = 3

# --- Configuration ---
# Choose a YOLOv8 model size. 'n' for nano (fastest, least accurate),
# 's' for small, 'm' for medium, 'l' for large, 'x' for extra large (slowest, most accurate).
# Start with 'n' or 's' if you're unsure or have limited GPU memory.
MODEL_SIZE = 'n' # Example: 'n' for YOLOv8 nano

# Training parameters
EPOCHS = 50       # Number of training epochs. Adjust as needed.
IMG_SIZE = 640    # Image size for training. Common values are 640, 1280.
BATCH_SIZE = 16   # Adjust based on your GPU memory. Smaller if you get OOM errors.
PROJECT_NAME = 'yolov8_mars_landing' # Name for your training run results folder

# --- 1. Download Dataset from Roboflow ---
print("--- Downloading dataset from Roboflow ---")
try:
    rf = roboflow.Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    data_yaml_path = os.path.join(dataset.location, 'data.yaml')
    print(f"data.yaml path: {data_yaml_path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please double-check your API_KEY, WORKSPACE, PROJECT, and VERSION.")
    exit() # Exit if dataset download fails

# --- 2. Load YOLOv8 Model ---
print(f"\n--- Loading YOLOv8{MODEL_SIZE}.pt model ---")
# 'yolov8n.pt' for nano, 'yolov8s.pt' for small, etc.
model = YOLO(f'yolov8{MODEL_SIZE}.pt')

# --- 3. Train the Model ---
print("\n--- Starting model training ---")
results = model.train(
    data=data_yaml_path,      # Path to your dataset's data.yaml
    epochs=EPOCHS,            # Number of epochs
    imgsz=IMG_SIZE,           # Image size
    batch=BATCH_SIZE,         # Batch size
    project=PROJECT_NAME,     # Project name for saving results
    name='train',             # Name for this specific run
    device=0                  # Use GPU (0 for the first GPU). Change to 'cpu' if no GPU.
)

# --- 4. Evaluate the Model (automatically done during training, but we can re-evaluate for specific metrics) ---
print("\n--- Evaluating the trained model ---")
# The 'train' function already calculates these metrics.
# We can access them from the 'results' object or re-run 'model.val()' for a dedicated validation.
metrics = model.val()

# Accessing metrics from the validation results
# mAP50-95 is the standard metric. mAP50 is also common for object detection.
# Recall and Precision are per-class, so we often look at the averaged values.
print(f"\n--- Training Results ---")
print(f"Mean Average Precision (mAP)@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"Mean Average Precision (mAP)@0.5:0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")

# Calculate F1-Score
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
# It's important to note that these are usually averaged metrics.
# For a more detailed F1-score, you'd calculate it per-class and then average.
# Ultralytics typically provides macro-averaged P, R, mAP.
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
if (precision + recall) != 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("F1-Score: Cannot be calculated (Precision + Recall is zero)")


# --- 5. Save the Model ---
# The trained model is automatically saved in the 'runs/detect/train' directory (or your custom project/name directory)
# as 'weights/best.pt'. We can also export it to a specific path for sharing.

# Path where the best model weights are saved by default
# This path is relative to where your script is run
runs_dir = os.path.join('runs', 'detect', 'train')
best_model_path = os.path.join(runs_dir, 'weights', 'best.pt')

if os.path.exists(best_model_path):
    print(f"\n--- Model Saved ---")
    print(f"Best model weights saved at: {best_model_path}")

    # You can also export the model to other formats (e.g., ONNX, OpenVINO, TFLite) for deployment
    # For sharing, 'best.pt' is usually sufficient.
    # If you want to explicitly save it to a different location:
    # model.save('my_mars_yolov8_model.pt')
    # print("Model also explicitly saved as 'my_mars_yolov8_model.pt'")
else:
    print("\n--- Model Not Found ---")
    print(f"Could not find the best model weights at: {best_model_path}")
    print("Please check the 'runs/detect/train' directory after training completes.")

print("\n--- Training complete! ---")