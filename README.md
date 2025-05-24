# Dental DICOM X-ray Analysis Web Application

## Overview
A full-stack web application that analyzes dental X-rays using AI-powered pathology detection and generates diagnostic reports. Built with FastAPI (Python) backend and React frontend.

## Features
- 🦷 Upload dental DICOM X-rays and images
- 🔍 AI-powered pathology detection using Roboflow
- 🎯 Visual annotation with bounding boxes
- 📊 Diagnostic report generation using Gemini LLM
- 📱 Responsive 2-panel dashboard interface
- ⚡ Real-time analysis and results

## Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.11.0
- **AI Integration**: Roboflow API for object detection
- **LLM**: Google Gemini for diagnostic reports
- **Image Processing**: PIL for annotation and manipulation

### Frontend (React)
- **Framework**: React 18+ with hooks
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **File Upload**: Drag & drop interface
- **Responsive**: Mobile-friendly design

## Setup Instructions

### Code setup (locally)
```
bash
git clone https://github.com/N-45div/Assignment.git
cd Assignment
```
### Docker build

```
docker-compose up --build
```

### Create Virtual Environment

```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running backend tests 

```
pytest test_main.py --cov=main --cov-report=html -v
```
![image](https://github.com/user-attachments/assets/bc7b3db0-f84c-4502-a6d4-5bb32b15e242)
