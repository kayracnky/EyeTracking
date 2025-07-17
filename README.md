# Eye Tracking Grid Navigation System

A computer vision-based eye tracking system designed for bedridden patients to navigate and interact with a 3x3 grid interface using only eye movements and blinks. This assistive technology enables hands-free communication and control for individuals with limited mobility.

## Project Overview

This system combines real-time eye tracking with UDP communication to provide an accessible interface for bedridden patients. Users can navigate through a grid of options using gaze direction and confirm selections through prolonged eye blinks.

### Key Features

- **Real-time Eye Tracking**: Uses MediaPipe for accurate gaze direction detection
- **Blink Detection**: Prolonged blinks (1.2+ seconds) for selection confirmation
- **3x3 Grid Navigation**: Intuitive grid-based interface with 9 selectable areas
- **Cancel Functionality**: Top-right corner serves as a cancel button to return to start
- **UDP Communication**: Real-time data transmission between Python tracker and Unity interface
- **Movement Cooldown**: Prevents accidental rapid movements (0.8 seconds between moves)
- **Visual Feedback**: Clear highlight indication and confirmation messages

## System Architecture

```
[Camera] → [Python Eye Tracker] → [UDP Socket] → [Unity Interface] → [User Feedback]
```

### Components

1. **tracking.py**: Standalone eye tracking without network communication
2. **trackingUDP.py**: Eye tracker with UDP transmission to Unity
3. **EyeTrackingReceiver.cs**: Unity script for receiving and processing eye data

## Requirements

### Software Dependencies

```bash
# Python packages
pip install opencv-python mediapipe numpy

# Unity
Unity 2021.3 LTS or newer
```

### Hardware Requirements

- **Camera**: Any USB webcam or built-in camera
- **Processing**: Minimum Intel i5 or equivalent (real-time processing)
- **RAM**: 4GB minimum, 8GB recommended
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+

## Installation & Setup

### 1. Python Environment Setup

```bash
# Clone the repository
git clone [your-repository-url]
cd eye-tracking-grid-navigation

# Install dependencies
pip install opencv-python mediapipe numpy

# Test standalone tracking
python tracking.py
```

### 2. Unity Project Setup

1. Open Unity and create a new 2D project
2. Import the `EyeTrackingReceiver.cs` script
3. Create the UI elements:
   - **Canvas**: Screen Space - Overlay mode
   - **Selection Highlight**: RectTransform for visual feedback
   - **Confirmation Text**: GameObject for selection feedback
   - **3x3 Grid Buttons** (optional): Named as "Button_0_0" to "Button_2_2"

### 3. Unity Configuration

```csharp
// Attach EyeTrackingReceiver.cs to a GameObject
// Configure in Inspector:
- Selection Highlight: Assign RectTransform
- Confirmation Text: Assign GameObject
- Button Width: 400px
- Button Height: 250px
- Button Spacing: 30px
- Movement Cooldown: 0.8 seconds
- Blink Threshold: 1.2 seconds
```

## Usage Instructions

### For Developers

1. **Start Unity Project**: Play the scene with EyeTrackingReceiver
2. **Run Eye Tracker**: Execute `python trackingUDP.py`
3. **Monitor Connection**: Check Unity console for UDP status messages

### For End Users (Patients)

1. **Position yourself**: Sit comfortably in front of the camera
2. **Calibration**: Look at different areas to test tracking accuracy
3. **Navigation**: 
   - Look **LEFT/RIGHT** to move horizontally
   - Look **UP/DOWN** to move vertically
4. **Selection**: **Blink and hold** for 1.2+ seconds to confirm
5. **Cancel**: Navigate to top-right corner and blink to return to start

## Grid Layout

```
Grid[0,0]     Grid[0,1]     Grid[0,2]
(-430,280)    (0,280)       (430,280)     ← CANCEL BUTTON
                                         
Grid[1,0]     Grid[1,1]     Grid[1,2]
(-430,0)      (0,0)         (430,0)      

Grid[2,0]     Grid[2,1]     Grid[2,2]
(-430,-280)   (0,-280)      (430,-280)    ← START POSITION
```

## Technical Specifications

### Eye Tracking Parameters

```python
# MediaPipe Configuration
max_num_faces = 1
refine_landmarks = True
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Gaze Detection Thresholds
horizontal_threshold = 0.15
vertical_threshold = 0.1

# Blink Detection
blink_threshold = 0.25  # EAR threshold
```

### Unity Grid System

```csharp
// Grid Dimensions
buttonWidth = 400f;      // pixels
buttonHeight = 250f;     // pixels
buttonSpacing = 30f;     // pixels

// Movement Settings
movementCooldown = 0.8f;  // seconds
blinkThreshold = 1.2f;    // seconds

// Network
UDP_PORT = 5053;
SERVER_IP = "127.0.0.1";
```

### UDP Protocol

**Data Format**: `GAZE:direction,BLINK:boolean`

**Examples**:
- `GAZE:LEFT,BLINK:false`
- `GAZE:RIGHT,BLINK:true`
- `GAZE:CENTER,BLINK:false`

**Directions**: LEFT, RIGHT, UP, DOWN, CENTER

## Troubleshooting

### Common Issues

1. **No UDP Connection**
   ```
   Solution: Check firewall settings, ensure port 5053 is available
   Debug: Monitor Unity console for UDP status messages
   ```

2. **Inaccurate Eye Tracking**
   ```
   Solution: Improve lighting, adjust camera position, clean camera lens
   Debug: Check MediaPipe face detection in Python window
   ```

3. **Delayed Response**
   ```
   Solution: Close other applications, check system resources
   Debug: Monitor frame rate in Python console
   ```

4. **Grid Misalignment**
   ```
   Solution: Verify Unity Canvas settings (Screen Space - Overlay)
   Debug: Check grid position calculations in Unity console
   ```

### Debug Information

**Python Console Output**:
- Face detection status
- Gaze direction calculations
- UDP transmission confirmations

**Unity Console Output**:
- Grid position updates
- Blink detection events
- Movement confirmations
- Cancel/selection actions

## Customization Options

### Adjusting Sensitivity

```python
# In trackingUDP.py - Modify thresholds
horizontal_threshold = 0.15  # Lower = more sensitive
vertical_threshold = 0.1     # Lower = more sensitive
blink_threshold = 0.25       # Lower = easier blink detection
```

### Unity Timing Settings

```csharp
// In EyeTrackingReceiver.cs Inspector
movementCooldown = 0.8f;  // Decrease for faster movement
blinkThreshold = 1.2f;    // Decrease for quicker selection
```

### Grid Customization

```csharp
// Modify grid dimensions
buttonWidth = 400f;   // Adjust button size
buttonHeight = 250f;  // Adjust button size
buttonSpacing = 30f;  // Adjust spacing between buttons
```

## Performance Optimization

### Python Optimization
- Use dedicated GPU for MediaPipe processing
- Reduce camera resolution if needed: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
- Close unnecessary applications

### Unity Optimization
- Use Release build for deployment
- Disable unnecessary Unity features
- Optimize UI rendering settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with real users if possible
4. Submit pull request with detailed description

### Areas for Improvement
- Calibration system for individual users
- Multiple language support
- Voice feedback integration
- Machine learning-based gaze prediction
- Mobile platform support

## License

This project is designed for assistive technology use. Please ensure compliance with medical device regulations in your jurisdiction when deploying for patient care.

## Support

For technical support or questions about implementation for medical environments, please create an issue in the repository.

## References

- [MediaPipe Face Mesh](https://mediapipe.dev/solutions/face_mesh)
- [Unity UDP Networking](https://docs.unity3d.com/Manual/UNet.html)
- [Eye Aspect Ratio for Blink Detection](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)

---

**Note**: This system is intended for assistive purposes. Always test thoroughly with target users and consider consulting healthcare professionals for medical applications.
