
---
# *<center>Deep Learning Based Face Recognition</center>*

*In this project, deep learning-based face recognition will be implemented on the
[**Labeled Face in the Wild**](http://vis-www.cs.umass.edu/lfw/ "Official Website") dataset.*

### Note: Work on this project is being continued
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
5. **Preparation5:** Fifth Data Preparation Process. 
In this notebook file 
a part that can be considered balanced was taken from the available dataset.
And an imbalanced part was taken by making it balanced from the available dataset.
Two new datasets named Between80And90 and Above90 were created.
Sample images were drawn from Training Validation and Test data.
Reorganized data was saved for future use.
See <ins>_/DataPreparation/Preparation5.ipynb_</ins> file for details.

### - Cascade

1. **haarcascade_frontalface_alt2:** xml file containing pre-trained model 
provided by _**opencv**_ library for face detection.
See [**Cascading Classifier Tutorial**](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html "docs.opencv") for more information.
See [**haarcascades File**](https://github.com/opencv/opencv/tree/master/data/haarcascades "github") published by _**opencv**_ for more cascading classifier models.
See <ins>_/Cascade/haarcascade_frontalface_alt2.xml_</ins> file for the classifier used in this project.

---
## *Training*

*First of all, I personally belive that it is necessary to understand the basis and philosophy of the method to be applied.
Simulating the real world in digital environment is the basis of computer science.
Many methods used in computer science are inspired by nature.
The Face Recognition(Image Classification) process that will be carried out within the scope of this project too will be simulate the image processing mechanism of the human brain.*

*Everything starts with how question.
How the human brain processes images?
Is the image (pixel values) taken from the human eyes really meaningful?
If a person with her/his eyes closed is given the sequential pixel values of an object, can the person recognize this object?
Or can pixel values used to describe it to someone who doesn't know what a pineapple looks like?
Sounds pretty silly doesn't it?
Things to mention instinctively to describe an object are its shape, color, texture, size etc.
We describe objects this way because that's how we actually see them and store them in our memory, not pixel values.
These are called [**Features**](https://en.wikipedia.org/wiki/Feature_(computer_vision) "wikipedia") of images or objects.
See also [**Features(Machine Learning)**](https://en.wikipedia.org/wiki/Feature_(machine_learning) "wikipedia").*

*The image processing section of the human brain is called [**Visual Cortex**](https://en.wikipedia.org/wiki/Visual_cortex "wikipedia").
If a little research is done about the Visual Cortex, it will be noticed that the image taken from the eyes undergoes many processes in Visual Cortex.
These processes can be described as [**Feature Extraction**](https://en.wikipedia.org/wiki/Feature_extraction "wikipedia") in its simplest form.
That is, the human brain processes images taken from the eyes by extracting their features.*

*[**Artificial Neural Networks(ANN)**](https://en.wikipedia.org/wiki/Artificial_neural_network "wikipedia") can be thought of as a virtual simulation of the human brain, they are frequently used in machine learning applications.
This structure has also been tried to be used for Image [**Classification**](https://en.wikipedia.org/wiki/Classification "wikipedia").
No success was achieved when a [**Feedforward Neural Network**](https://en.wikipedia.org/wiki/Feedforward_neural_network "wikipedia") was trained with image pixels.
However, successful results were achieved when the training was repeated with the features extracted from the images.
In the beginning, these features were extracted with some algorithms but this was inefficient in many ways.
First of all, these algorithms could've taken a long time to work.
In addition, the features to be extracted had to be determined manually.
This could result in ignoring many features that are valuable for the relevant classification.*

*This process continued until the [**Convolutional Neural Network(CNN)**](https://en.wikipedia.org/wiki/Convolutional_neural_network "wikipedia"), which made feature extraction a part of the learning process, which was developed by exploring the Visual Cortex at a simulable level.
See also [**Convolutional Neural Network(CNN)**](https://www.ibm.com/cloud/learn/convolutional-neural-networks "IBM"). 
Check [**This Article**](https://arxiv.org/ftp/arxiv/papers/2001/2001.07092.pdf "arxiv") for the relationship between Visual Cortex and Convolutional Neural Network(CNN).*

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