# Object Detection and Tracking
### Object detection and tracking using Mask RCNN and temporal coherence 
This is the implementation of manufacturing Object detection and tracking in the manufacturing plants. This model uses Mask RCNN model to do the initial segmentation. Which is based on Feature Pyramid Network(FPN) and a ResNet50 backbone. To give temporal consistency in the detection results, a two-staged detection threshold has been used to boost up weak detections in a frame by referring to objects with high detection scores in neighboring frames.


![ezgif com-crop (1)](https://user-images.githubusercontent.com/40798690/57718148-4a6bab00-7642-11e9-9903-309df505b236.gif)

The repository includes:
* Source code of Mask R-CNN with temporal coherence
* Training code
* Jupyter notebook for detection
* Evaluation file for precision, recall and F1-score

## Getting Started
* Install the required dependencies: (for reference see [how_to_install.pdf](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/how_to_install.pdf) )
* [custom.py](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/custom.py) : this code is used for loading data and training the model
* [Training.ipynb](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/Training.ipynb): loading the weight and calling the training function
* [result_calculation.ipynb](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/result_calculation.ipynb): this code is used for detecting objects with or without temporal coherence. This also calculates precision, recall and f1-score of the model.
* [mrcnn/visualize_frame_relation_4f.py](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/mrcnn/visualize_frame_relation_4f.py) : this code is used for visualizing the detected objects with mask.

## Training on your dataset
Use the [custom.py](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/custom.py) to set your number of classes and load the dataset (train, val) with annotation. For annotating the images use COCO style image annotation tool, which are available online.

## Citation
If you use this repository, please cite the following paper:

~~~~
@inproceedings{Kari1908region,
  author ={Karim, Muhammad Monjurul and Doell, David and Lingard, Ravon and Yin, Zhaozheng and Leu, Ming and Qin, Ruwen},
  title={A {Region-Based} Deep Learning Algorithm for Detecting and Tracking Objects
  in Manufacturing Plants},
  booktitle ={25th International Conference on Production Research 2019 (ICPR 2019)},
  ADDRESS={Chicago, USA},
  YEAR={2019}
}
~~~~

