
---
# *<center>Deep Learning Based Face Recognition</center>*

*In this project, deep learning-based face recognition will be implemented on the
[**Labeled Face in the Wild**](http://vis-www.cs.umass.edu/lfw/ "Official Website") dataset.*

### Note: Work on this project continues
---

### *Dataset*
*In this context, raw images called
[**All Images**](http://vis-www.cs.umass.edu/lfw/lfw.tgz "tgz File Link")
of the 
[**Labeled Face in the Wild**](http://vis-www.cs.umass.edu/lfw/ "Official Website") dataset will be used.
Also see [**Kaggle Link**](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset "Kaggle Link") of dataset.*

---

### *Version*

- _Python Version: **3.9.12**_
- _Numpy Version: **1.22.3**_
- _OpenCV(cv2) Version: **4.5.1**_
- _Pandas Version: **1.4.3**_
- _Matplotlib Version: **3.5.2**_

---

# *<center>Notebooks</center>*

---

## *Data Preparation*

### - Preparation

1. **Preparation1:** First Step of data preparation process. 
In this notebook file image data was read from raw image files with _**opencv**_ library,
dataset was cleaned from non-quality redundancy and shrinked.
Sample images were drawn from Training Validation and Test data.
Dataset was splitted into Training, Validation and Test data and IDs were resetted.
Performing the splitting process in the first stage will allow all models to be trained with the same Training data.
Thus, the performance change caused by the Training data difference will be prevented.
In this way, it will be possible to determine more accurately which models and parameters are better.
Selected data were saved as pkl file for future use.
See <ins>_/DataPreparation/Preparation1.ipynb_</ins> file for details.
2. **Preparation2:** Second Data Preparation Process. 
In this notebook file previously selected and saved data was read from pkl file, 
Training, Validation and Test images were resized to (224, 224, 3).
The reason why images were resized this way will be explained in the Training section.
Sample images were drawn from Training Validation and Test data.
Reorganized data were saved as pkl file for future use.
See <ins>_/DataPreparation/Preparation2.ipynb_</ins> file for details.
3. **Preparation3:** Third Data Preparation Process. 
In this notebook file previously selected and saved data was read from pkl file.
Faces in all images of selected data were detected
via a [**Cascading Classifier**](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html "docs.opencv") model.
Images in DataFrames were replaced with images containing extracted faces.
Detection types of extracted faces were appended relevant DataFrames.
Sample images were drawn from Training Validation and Test data.
Reorganized data was saved for future use.
This project will not focus on face detection.
For this reason, the details of face detection will not be mentioned.
See [**Face Detection**](https://en.wikipedia.org/wiki/Face_detection "wikipedia") for more information.
See <ins>_/DataPreparation/Preparation3.ipynb_</ins> file for details.
4. **Preparation4:** Fourth Data Preparation Process. 
In this notebook file 
two parts that can be considered balanced were taken from the available dataset.
Two new datasets named FirstQuarter and ThirdQuarter were created, 
although they do not exactly correspond to a quarter of the number of people in the dataset.
Sample images were drawn from Training Validation and Test data.
Reorganized data was saved for future use.
See <ins>_/DataPreparation/Preparation4.ipynb_</ins> file for details.

### - Cascade

1. **haarcascade_frontalface_alt2:** xml file containing pre-trained model 
provided by _**opencv**_ library for face detection.
See [**Cascading Classifier Tutorial**](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html "docs.opencv") for more information.
See [**haarcascades File**](https://github.com/opencv/opencv/tree/master/data/haarcascades "github") published by _**opencv**_ for more cascading classifier models.
See <ins>_/Cascade/haarcascade_frontalface_alt2.xml_</ins> file for the classifier used in this project.

---

- _**Note:** All data files are deleted with their folders to save memory.
If notebook files are wanted to be run, all folders in the lfw folder in the 
[**lfw.tgz**](http://vis-www.cs.umass.edu/lfw/lfw.tgz "tgz File Link")
file must be copied into the
<ins>/Data/RawData/FaceImage/</ins> 
path.
Non-existent folders must be created and notebook files must be run in the specified order._

---

## <center>_I'd Be Glad If You Report Any Mistakes You Notice._</center>

---