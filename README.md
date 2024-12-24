# AeroGaze

AeroGaze is a project developed as part of **Aerothan 2024**, with a focus on **surveillance and disaster management**. The repository contains models and scripts that enable detection, classification, and precision landing on a bullseye, providing a robust solution for autonomous drone operations in critical scenarios.

---

## **Features**

### 1. **Detection and Classification**
- Implements state-of-the-art object detection models to identify and classify relevant entities in the environment.

### 2. **Precision Landing**
- Utilizes a model tailored for accurate drone landing on a predefined bullseye marker.
- Ensures high precision for payload delivery and autonomous drone navigation.

### 3. **Surveillance and Disaster Management**
- Designed to aid in real-time monitoring of disaster-affected areas.
- Facilitates effective decision-making through advanced AI-driven analytics.

---

## **Repository Structure**

### **1. Main Missions**
Contains video and mission data:
- Videos: Demonstrations of various mission scenarios.

### **2. Models**
- Pre-trained models for detection and classification.
  - `best.pt`: High-accuracy detection model.
  - `bullseye.pt`: Model for precision landing.

### **3. Scripts**
- Core functionality scripts:
  - `bytetracker.py`: Tracks objects in real time.
  - `counter.py` and `counters.py`: Count detected entities.
  - `live.py`: Live detection and analysis.
  - `mission1.py`: Executes Mission 1 sequence.
  - `pinkish_to_normal.py`: Image preprocessing.
  - `scrapper.py`: Data collection utility.
  - `sobel.py` and `sobel_scrapper.py`: Edge detection scripts.

### **4. Raw Images**
- Contains example images for model training and validation.

---

## **Getting Started**

### **Prerequisites**
- Python 3.8+
- PyTorch
- OpenCV
- Required Python packages listed in `requirements.txt`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/jayaprakashll/AeroGaze.git
   cd AeroGaze
