# Real-time Sign Language Translator

A computer vision application that translates hand gestures into spoken words in real-time using webcam input. This project helps bridge communication gaps between sign language users and those who don't understand sign language.

## 🎓 Academic Project

This is a final year project for Electronics and Computer Engineering at Nnamdi Azikiwe University, demonstrating the practical application of computer vision and machine learning for accessibility.

## 🎯 What It Does

- **Real-time Hand Gesture Recognition**: Detects and tracks hand movements using your webcam
- **Sign Language Translation**: Converts recognized signs into spoken words
- **Live Feedback**: Provides visual and audio output for immediate communication
- **Accessibility Focus**: Designed to help hearing-impaired individuals communicate more easily

## 🚀 How It Works

1. **Hand Detection**: Uses MediaPipe to track 21 key points on your hand
2. **Gesture Recognition**: Neural network classifies hand positions into sign language
3. **Text-to-Speech**: Converts recognized signs into spoken words
4. **Real-time Processing**: Processes video feed at 30+ FPS for smooth interaction

## 📋 Supported Signs

Currently recognizes 5 basic signs:

- **Wait** - Hold gesture
- **Thank you** - Gratitude gesture
- **You** - Pointing gesture
- **Okay** - Thumbs up gesture

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7+
- Webcam
- Microphone (for text-to-speech)

### Setup Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
python main.py
```

### Terminate the Application

- **ESC**: Exit application (while application window is active)
- **Ctrl + C**: Force stop in terminal
- **Cmd + Q**: Quit application window (macOS)

## 🎮 Usage

### Basic Operation

1. Run the application
2. Show your hand to the webcam
3. Perform one of the supported signs
4. The system will speak the recognized word

### Controls

- **ESC**: Exit application

### Command Line Options

```bash
python main.py --device 0 --width 640 --height 360
```

## 🧠 Machine Learning Model

- **Input**: 42 features (21 hand landmarks × 2 coordinates)
- **Architecture**: Neural network with dropout layers for robustness
- **Training**: Custom dataset with 5 sign classes
- **Optimization**: TensorFlow Lite for fast inference

## 📁 Project Structure

```
realtime-sign-language-translator/
├── main.py                          # Main application
├── modules.py                       # Core functions
├── model/                          # ML models
│   └── keypoint_classifier/        # Hand gesture classifier
├── utils/                          # Utility functions
├── docs/                           # Documentation
└── keypoint_classification_EN.ipynb # Model training notebook
```

## 🔧 Customization

### Adding New Signs

1. Collect training data using the logging modes
2. Retrain the model using the Jupyter notebook
3. Update the label file with new signs

### Adjusting Performance

- Modify detection confidence thresholds
- Adjust camera resolution settings
- Fine-tune the neural network architecture

## 🔧 Troubleshooting

### Camera Permission Issues

- Grant camera permissions when prompted
- On macOS, go to System Preferences > Security & Privacy > Camera

### Text-to-Speech Issues

- The application uses macOS built-in `say` command for text-to-speech
- If you hear no audio, check your system volume and audio settings

### Virtual Environment Issues

- Make sure you're in the project directory when creating the virtual environment
- Always activate the virtual environment before running the application
- If you get import errors, reinstall dependencies: `pip install -r requirements.txt`

**Built with ❤️ for accessibility and inclusion**
