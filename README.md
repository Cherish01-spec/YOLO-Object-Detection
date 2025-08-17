# YOLO Object Detection App

This is a **Python Tkinter application** for real-time **object detection** using the **YOLOv3** deep learning model.  
It allows you to:
- Detect objects in **uploaded images**
- Detect objects in **uploaded videos**
- Detect objects **live** through your laptop camera

---

## 📌 Features
- **YOLOv3** deep learning model for object detection
- Image and video upload support
- Live camera feed detection
- Full-screen responsive interface
- Displays **total object count** and **detailed counts per object**
- Easy to exit with a dedicated **Exit** button

---

## 📂 Project Structure
```
yolo_object_detection/
│
├── app.py             # Main application file
├── yolov3.cfg         # YOLOv3 model configuration
├── yolov3.weights     # YOLOv3 pretrained weights
├── coco.names         # Object class names
├── README.md          # Project documentation
└── env/               # Python virtual environment (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://https://github.com/Cherish01-spec/YOLO-Object-Detection.git
cd YOLO-Object-Detection
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv env
.\env\Scripts\activate     # For Windows
# source env/bin/activate  # For Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, install manually:
```bash
pip install opencv-python pillow numpy
```

### 4️⃣ Download YOLO Files
- **yolov3.weights**
- **yolov3.cfg**
- **coco.names**

Place these files in the same directory as `app.py`.

---

## ▶️ Running the Application
```bash
python app.py
```

---

## 🖼️ Usage
1. **Upload Image** → Detects objects in selected image file.
2. **Upload Video** → Detects objects in a video file.
3. **Live Video** → Detects objects in real-time using laptop camera.
4. **Exit** → Closes the application.

---

## 📌 Requirements
- Python 3.7+
- OpenCV
- Pillow
- NumPy
- YOLOv3 Model files (`yolov3.cfg`, `yolov3.weights`, `coco.names`)

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify.
