# Chicken Detection and Counting with YOLO Pretrained Model
Detect chicken escaping our backyard into our front yard and send a notification to an application.

# To-do List
- Implement it on SCB
- Capture data from live feed RTSP, ESP32CAM
- Publish live info collected to an application, probably web.
- Utilize better model, since Yolo detect "birds"
- Maybe train a model to detect each chicken by its name, like [here](https://github.com/DennisFaucher/ChickenDetection).

# Current challenges
- 2 chickens are so similar that the model struggles to differentiate them. 
- These 2 chickens can and are probably being differentiated by their combs, but the color of the combs are "seasonal", when chickens are on their laying eggs season theirs combs are usually bright red, and grow rosier when broody.



## Introduction
This project aims to detect and count chickens using the YOLO (You Only Look Once) pretrained model. The purpose is to monitor chicken enclosures and detect any instances of chickens attempting to escape.

## Prerequisites
- Python 3.x
- OpenCV
- YOLO Pretrained Model
- numpy
- ultralytics
- cvzone
- scikit-image
- filterpy
```diff
- Add remaining prerequisites
```


## Installation
1. Clone the repository:

```
git clone 
```
1. Install dependencies:

```
pip install -r requirements.txt
```



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
