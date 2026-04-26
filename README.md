# Flood Rescue Detection System

This is a drone-camera rescue alert system designed to process aerial footage and detect `humans`, `cars`, `humans in cars`, and `drowning` situations using a custom YOLO model. It provides real-time detection tallies, confidence threshold tuning, and critical emergency alerts for life-saving operations during floods.

## Features
*   **Video & Image Analysis**: Upload standard images or video footage (`.mp4`, `.avi`, `.mov`) directly to the interface.
*   **Custom YOLO Inference**: Uses `best.pt` object detection weights for classes: `car`, `person`, `in-car`, and `drowning`.
*   **Real-time Analytics**: Displays live metrics of how many objects were found per frame.
*   **Emergency Alerting System**: Dynamically flashes prominent alert banners to the operator if a `drowning` situation or `in-car` detection occurs.
*   **Live Parameter Tuning**: Use the sidebar slider to adjust the model's confidence threshold to filter out blurred drone footage.

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd Flood_rescue_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your custom weights file (`best.pt`) is in the root directory.

## Run the Application

```bash
streamlit run app.py
```
Open the provided `localhost` link in your browser to view the dashboard!
