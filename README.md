
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
- _Tensorflow Version: **2.6.0**_
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

*Considering human experiences on this subject, visual cortex does not develop with a limited number of visuals in a limited field.
The human sees and analyzes images continuously as long as the eyes are open.
The brain, which is constantly changing, continues to improve itself by learning something new from each image.
In this way, the knowledge learned from experience in any field can be used in other fields if it is meaningful.
This is the motivation for [**Transfer Learning and Fine-Tuning**](https://www.tensorflow.org/tutorials/images/transfer_learning "tensorflow").
See also [**Transfer Learning (wikipedia)**](https://en.wikipedia.org/wiki/Transfer_learning "wikipedia")
and [**Fine-Tuning (deeplizard)**](https://deeplizard.com/learn/video/5T-iXNNiwIs "deeplizard").*

*In this section, different models will be trained with the prepared datasets and the results will be shown.
The models to be trained will be based on the architectures of the [**VGG16**](https://keras.io/api/applications/vgg/ "keras") and [**MobileNet**](https://keras.io/api/applications/mobilenet/#mobilenet-function "keras") pre-trained models.
See also [**MobileNet (GitHub)**](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md "github") and [**Depthwise Separable Convolutions**](https://towardsdatascience.com/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503 "towardsdatascience").
See also [**VGG16 (TensorFlow)**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16 "tensorflow")*

*Many pre-trained models, including the ones to be used within the scope of this project, have been trained with (224, 224, 3) sized images containing pixel values in the [-1, 1] range.
This is why the images were resized as (224, 224, 3) in the Preparation section.
Also, for this reason, the pixel values will be converted to the range [-1, 1].
In this way, the data will be symmetrical and the performance of the [**Backpropagation**](https://en.wikipedia.org/wiki/Backpropagation "wikipedia") algorithm used during training will be increased.
See also [**This Question and Answer**](https://stackoverflow.com/questions/59540276/why-in-preprocessing-image-data-we-need-to-do-zero-centered-data "stackoverflow").
Therefore, training will be performed by converting pixel values to this range with the simplest method (pixel / 127.5 - 1).*

*The same trainings will be carried out by applying [**Data Augmentation**](https://en.wikipedia.org/wiki/Data_augmentation "wikipedia") to the data in order to observe the difference.
For details of the Data Augmentation process used in this application, see also [**Image Data Augmentation**](https://www.tensorflow.org/tutorials/images/data_augmentation "tensorflow").*

### - ResizedData - FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNet architecture were trained with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, a model based on MobileNet architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData-FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData-FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training12.ipynb_</ins> file for details.




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