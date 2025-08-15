import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from collections import Counter

# Load YOLO Model
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
coco_names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize Tkinter window (Full-Screen)
root = tk.Tk()
root.title("YOLO Object Detection")
root.attributes("-fullscreen", True)  # Full-screen mode
root.configure(bg="black")

# Labels for total object count
count_label = Label(root, text="Total Objects: 0", font=("Arial", 16, "bold"), fg="white", bg="black")
count_label.pack(pady=10)
details_label = Label(root, text="", font=("Arial", 14), fg="white", bg="black")  # Specific object count
details_label.pack(pady=5)

# Function to detect objects
def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            detected_objects.append(classes[class_ids[i]])  # Store detected object names
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Count occurrences of each detected object
    object_counts = dict(Counter(detected_objects))
    total_objects = sum(object_counts.values())

    # Format object count details
    details_text = ", ".join([f"{obj}: {count}" for obj, count in object_counts.items()])
    
    # Update labels
    count_label.config(text=f"Total Objects: {total_objects}")
    details_label.config(text=f"Detected: {details_text}")

    return image

# Function to process image
def process_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    if not filepath:
        return
    image = cv2.imread(filepath)
    image = detect_objects(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    label.config(image=image)
    label.image = image

# Function to process video
def process_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi")])
    if not filepath:
        return
    cap = cv2.VideoCapture(filepath)
    
    def play_video():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            label.config(image=frame)
            label.image = frame
            root.update_idletasks()
        
        cap.release()

    threading.Thread(target=play_video, daemon=True).start()

# Function to start live video detection using laptop camera
def live_video():
    cap = cv2.VideoCapture(0)  # Open the webcam

    def process_live():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            label.config(image=frame)
            label.image = frame
            root.update_idletasks()

        cap.release()

    threading.Thread(target=process_live, daemon=True).start()

# Function to exit the application
def exit_app():
    root.quit()

# UI Elements
btn_frame = tk.Frame(root, bg="black")
btn_frame.pack(pady=20)

btn_img = tk.Button(btn_frame, text="Upload Image", font=("Arial", 14, "bold"), fg="white", bg="red", command=process_image)
btn_img.pack(side=tk.LEFT, padx=10)

btn_vid = tk.Button(btn_frame, text="Upload Video", font=("Arial", 14, "bold"), fg="white", bg="red", command=process_video)
btn_vid.pack(side=tk.LEFT, padx=10)

btn_live = tk.Button(btn_frame, text="Live Video", font=("Arial", 14, "bold"), fg="white", bg="red", command=live_video)
btn_live.pack(side=tk.LEFT, padx=10)

btn_exit = tk.Button(btn_frame, text="Exit", font=("Arial", 14, "bold"), fg="white", bg="red", command=exit_app)
btn_exit.pack(side=tk.LEFT, padx=10)

label = Label(root, bg="black")
label.pack()

# Run Application
root.mainloop()