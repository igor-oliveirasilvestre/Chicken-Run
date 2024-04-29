# Chicken Detection and Counting with YOLO Pretrained Model

## Introduction
This project aims to detect and count chickens using the YOLO (You Only Look Once) pretrained model. The purpose is to monitor chicken enclosures and detect any instances of chickens attempting to escape.

## Prerequisites
- Python 3.x
- OpenCV
- YOLO Pretrained Model
<font color='red'>
- Add remaining prerequisites
 </font>


## Installation
1. Clone the repository:
2. Install dependencies:
<font color='red'>
3. make a requisites.txt file
 </font>


## Usage
1. Download the YOLO pretrained model weights and configuration files. You can find them [here](https://github.com/AlexeyAB/darknet) or use other YOLO models.
2. Place the downloaded files in the `model` directory.
3. Run the detection script:

```
python chickenrun.py
```

## Configuration
- `detect_chickens.py`: This script contains the main logic for detecting and counting chickens. You can adjust parameters such as confidence threshold and non-maximum suppression threshold in this file.
- `model/`: This directory contains the YOLO pretrained model weights and configuration files. You can replace them with other YOLO models if needed.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request with any improvements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- YOLO Pretrained Model: [YOLOv4](https://github.com/AlexeyAB/darknet)
- OpenCV Library: [OpenCV](https://opencv.org/)
