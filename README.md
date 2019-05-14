# Object Detection and Tracking
### Object detection and tracking using Mask RCNN and temporal coherence 
This is the implementation of manufacturing Object detection and tracking in the manufacturing plants. This model uses Mask RCNN model to do the initial segmentation. Which is based on Feature Pyramid Network(FPN) and a ResNet50 backbone. To give temporal consistency in the detection results, a two-staged detection threshold has been used to boost up weak detections in a frame by referring to objects with high detection scores in neighboring frames.


![ezgif com-crop (1)](https://user-images.githubusercontent.com/40798690/57718148-4a6bab00-7642-11e9-9903-309df505b236.gif)

The repository includes:
* Source code of Mask R-CNN with temporal coherence
* Training code
* Jupyter notebook for detection
* Evaluation file for precision, recall and F1-score

### Getting Started
* Install the required dependencies:

> pip install -r requirements.txt

* [custom.py](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/custom.py) : this code is used for loading data and training the model
* [Training.ipynb](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/Training.ipynb): loading the weight and calling the training function
* [result_calculation.ipynb] (https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/result_calculation.ipynb): this code is used for detecting objects with or without temporal coherence. This also calculates precision, recall and f1-score of the model.
* [mrcnn/visualize_frame_relation_4f.py] (https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/mrcnn/visualize_frame_relation_4f.py) : this code is used for visualizing the detected objects with mask.




