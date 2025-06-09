# 🚢 Ship Image Classification

## Business Context

**NauticAI Solutions** is a leader in developing technological tools for the maritime industry. One major challenge is managing port traffic data — particularly, logging ships that enter and exit various harbors.  
Currently, this is a **manual and time-consuming process**.

To improve efficiency, NauticAI seeks to automate ship recognition using a classification system based on camera-captured images at ports. These images include 10 ship types, such as:

- Cruise ships
- Cargo ships
- Tankers
- Sailboats
- Submarines
- Merchant ships
- Fishing boats
- Yachts
- Patrol boats
- Tugboats

### Why This Matters

- **Faster processing**: Real-time identification of ships  
- **Better accuracy**: Reduces human error in categorization  
- **Improved audits**: Creates a reproducible, trackable system for inspections  

---

## 📁 Project Structure

| File / Notebook                      | Description                                                                                                         |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `1.0-data-cleaning-and-eda.ipynb`   | Cleans the dataset and performs Exploratory Data Analysis (EDA). Checks image quality, class distribution, etc.     |
| `2.0-resnet-baseline-training.ipynb`| Trains a ResNet18 model as a baseline. Includes layer freezing, dropout, and sampling to reduce overfitting.        |
| `2.0_loss_logs.txt`                 | Training log file containing metrics like loss, accuracy, F1-score across epochs for the ResNet18 model.            |
| `3.0-yolov8-training.ipynb`         | Trains a YOLOv8 model for object detection using bounding boxes. Evaluates performance with mAP and loss curves.    |

---

## Insights from `2.0-resnet-baseline-training.ipynb`

### Hyperparameter Tuning Attempted for Overfitting

**Validation performance was poor and unstable**, showing low accuracy, inconsistent F1-score, and irregular precision/recall.

### Model Fails to Generalize

- Large gap between training and validation metrics across epochs.  
- Clear signs of memorization, not meaningful learning.

### Instability in Validation

- Sudden spikes in loss and drops in metrics.  
- Likely causes: noise or imbalance in validation data.

---

I've tried a few things to reduce overfitting:

- Made training and validation more balanced using `WeightedRandomSampler`  
- Made the last layer smaller (from 256 to 128 units)  
- Added Dropout (0.3) to help the model not depend too much on specific parts  
- Let layers 3 and 4 learn too, not just the last layer

> Even after all these changes, the model still struggled. It got really good at the training data, but couldn’t do as well when faced with new images.

### Decision: Switch to YOLOv8

Due to the generalization issues with ResNet18, we moved to a YOLOv8 object detection model.

---

## What Went Well With YOLO

- Generalized well — didn't just memorize training images.  
- Categories like **submarines**, **yachts**, and **fishing boats** were predicted **almost perfectly**.  
- Achieved **~0.80 mAP@0.5**, showing solid performance across classes.  
- Training was stable with consistent metric improvement.  
- **Box loss could be lower** — predictions were sometimes off in size or placement.

---

## Test Set Results Summary

- 100% accuracy on Fishing Boat, Merchant Ship, and Sails Boat  
- High performance on Yacht (89%), Submarine (88%), and Patrol Boat (90%)  
- Military Ship had some confusion with nearby classes 
- Tugboat had no correct predictions and very few samples (only 42 Bounding boxes) — it may be worth removing it or do oversampling to improve this class

➡️ These results confirm the model performs reliably on unseen data, and generalizes better than the older architecture.

---

## Real-Time Use at Ports

### Integration Plan

- **Hardware**
  - High-resolution surveillance cameras (day/night capable)
  - An edge computing device (e.g. NVIDIA Jetson, industrial GPU-powered unit) or server for inference

- **Software**
  - Inference pipeline with the trained YOLO model (via Ultralytics)
  - API or middleware to integrate detections with the port's logistics/registry system
  - Dashboard (e.g. Streamlit, Dash) for visual monitoring and manual override if needed

### Handling Unknown Ships
- The current model classifies only 10 predefined ship types.
- To manage **new or unknown ship types**, the system can:
  - Flag predictions with **low confidence** for manual review
  - Use an **"unknown" fallback label** when no class surpasses a confidence threshold (e.g., 0.3)
  - Log and store those images for **future model re-training** to incorporate emerging ship types

### Continuous Learning (Optional)
- Establish a feedback loop: operators can correct wrong predictions.
- Periodically retrain the model with newly labeled data to improve performance and adaptability.
