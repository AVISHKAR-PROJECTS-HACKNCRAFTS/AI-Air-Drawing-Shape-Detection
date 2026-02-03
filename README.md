# Air-Droid-vignan

AI Air Drawing Shape Detection

CNN + Vision Transformers | Streamlit | Python

📌 Project Overview

This project is an AI-powered Air Drawing application that allows users to draw shapes in the air using hand gestures.
The system captures hand movements through a camera, processes the gesture trajectory, and intelligently recognizes the drawn shape using Deep Learning (CNN + Vision Transformers).
Once detected, the shape is rendered cleanly on the screen with color filling.

The application is built using Python and Streamlit and follows a modular, client-ready architecture.

🎯 Key Features

✋ Air Drawing using Hand Gestures

🎥 Real-time Camera Input

🖐 Hand Detection & Fingertip Tracking

🧠 AI-based Shape Recognition (CNN + Vision Transformers)

🎨 Automatic Color-Filled Shape Rendering

📐 Supports 10+ Shapes

🧩 Modular & Scalable Architecture

🌐 Interactive Streamlit Web Interface

🧠 Supported Shapes

The system currently supports at least 10 shapes:

Circle

Square

Rectangle

Triangle

Pentagon

Hexagon

Heptagon

Octagon

Star

Ellipse

(Architecture allows easy extension to more shapes.)

🏗 System Architecture

The project strictly follows the below pipeline:

Real-Time Camera Input
        ↓
Hand Detection & Tracking (OpenCV)
        ↓
Image Preprocessing Pipeline
    - Background Removal
    - Frame Selection
    - Gesture Contour Detection
        ↓
CNN Feature Extraction
        ↓
Vision Transformer (Attention-Based Learning)
        ↓
Shape Classification
        ↓
Clean Shape Rendering with Color Fill (Streamlit UI)


This architecture ensures robust AI-based shape recognition, not rule-based detection.

🛠 Technologies Used
Core Technologies

Python – main programming language

Streamlit – interactive web UI

OpenCV – camera handling & image processing

AI & Deep Learning

Convolutional Neural Networks (CNN) – local feature extraction

Vision Transformers (ViT) – global shape understanding

TensorFlow / Keras or PyTorch – deep learning framework

🚀 How It Works (Simple Explanation)

The camera captures live video.

The system detects and tracks the user’s index fingertip.

Hand movement is recorded as a gesture trajectory.

The gesture is preprocessed into a clean stroke.

A CNN extracts spatial features from the stroke.

A Vision Transformer analyzes the global shape pattern.

The shape is classified and rendered as a perfect, color-filled shape on the canvas.

🖥 User Interface (Streamlit)

The application UI includes:

Live camera preview

Hand landmark visualization

Smooth drawing overlay

Start / Stop Camera controls

Clear Canvas

Undo Last Shape

Color selection

Side panel showing detected shapes

Clean and professional UI suitable for client demos
