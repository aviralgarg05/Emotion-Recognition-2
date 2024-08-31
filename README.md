
# Facial Emotion Recognition using OpenCV and DeepFace

This repository contains the implementation of a real-time facial emotion recognition system using OpenCV and DeepFace. The project is designed to recognize human emotions from facial expressions captured through a webcam in real-time. The system detects faces in the video stream, processes them, and classifies the emotions into categories like happiness, sadness, anger, and surprise.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Optimization](#optimization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Real-Time Emotion Detection**: Detects and classifies emotions in real-time using a webcam feed.
- **High Accuracy**: Utilizes DeepFace models which are pre-trained on large datasets for accurate emotion classification.
- **Optimized for Performance**: Balances between accurate detection and high frames per second (FPS) for a smooth real-time experience.
- **Multiple Detection Models**: Includes support for different `haarcascade` classifiers to enhance face detection accuracy.
- **Modular Codebase**: Easy to modify and extend for custom models or additional features.


## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/aviralgarg05/Emotion-Recognition-2.git
   cd Emotion-Recognition-2
   ```

2. **Install Dependencies:**

   Make sure you have Python installed. Then, install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install the dependencies manually:

   ```bash
   pip install opencv-python deepface
   ```

## Usage

1. **Run the Emotion Recognition:**

   Simply execute the script to start the emotion recognition system:

2. **Adjust Settings:**

   - **Face Detection**: Modify the `haarcascade` XML files in the code to try different face detection models.
   - **Model Tuning**: Experiment with different DeepFace models for improved emotion classification.

## Project Structure

The repository is organized as follows:

Emotion-Recognition-2/
├── main.py                # Main script for running the emotion recognition
├── requirements.txt       # List of dependencies
├── haarcascades/          # Directory containing haarcascade XML files
│   ├── haarcascade_frontalface_default.xml
│   └── ...
├── README.md              # Project documentation


## Optimization

- **Improving FPS**: The code is optimized for higher FPS by adjusting detection intervals and model processing times.
- **Alternative Libraries**: Consider using more lightweight face detection libraries if you need further improvements in performance.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- **OpenCV**: For providing an open-source computer vision library.
- **DeepFace**: For the pre-trained models and easy-to-use APIs for emotion recognition.
```
