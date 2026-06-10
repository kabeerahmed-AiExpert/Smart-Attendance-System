# Smart Attendance System using Face Recognition

## Overview
This project is an AI-based Smart Attendance System that uses Face Recognition and Computer Vision to automatically mark attendance in real time. It removes the need for manual attendance and improves accuracy, speed, and efficiency in classrooms and workplaces.

## Problem
Manual attendance systems are slow, time-consuming, and prone to proxy or human errors. Managing attendance for large groups becomes inefficient and unreliable.

## Solution
This system detects and recognizes faces using a camera and automatically records attendance when a match is found in the database.

## Features
- Real-time face detection and recognition  
- Automatic attendance marking  
- Face encoding and matching system  
- Attendance storage in CSV/log format  
- Dataset creation for training faces  
- Simple and user-friendly interface  

## Tech Stack
- Python  
- OpenCV  
- NumPy  
- Face Recognition Library  
- Pandas  

## Project Structure
Smart-Attendance-System/
├── dataset/
├── training/
├── attendance/
├── main.py
├── encode_faces.py
├── requirements.txt
└── README.md

## How to Run
```bash
git clone https://github.com/kabeerahmed-AiExpert/Smart-Attendance-System
cd Smart-Attendance-System
pip install -r requirements.txt
python main.py
