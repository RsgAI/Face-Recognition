
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

### - ResizedData / FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FullPhoto/Training12.ipynb_</ins> file for details.

### - ResizedData / FaceOnly

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the ResizedData/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on ResizeData/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ResizedData/FaceOnly/Training12.ipynb_</ins> file for details.

### - FirstQuarter / FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the FirstQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FullPhoto/Training12.ipynb_</ins> file for details.

### - FirstQuarter / FaceOnly

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the FirstQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on FirstQuarter/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/FirstQuarter/FaceOnly/Training12.ipynb_</ins> file for details.

### - ThirdQuarter / FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the ThirdQuarter/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FullPhoto/Training12.ipynb_</ins> file for details.

### - ThirdQuarter / FaceOnly

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the ThirdQuarter/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on ThirdQuarter/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/ThirdQuarter/FaceOnly/Training12.ipynb_</ins> file for details.

### - Between80And90 / FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the Between80And90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FullPhoto/Training12.ipynb_</ins> file for details.

### - Between80And90 / FaceOnly

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the Between80And90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on Between80And90/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Between80And90/FaceOnly/Training12.ipynb_</ins> file for details.

### - Above90 / FullPhoto

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the Above90/FullPhoto dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FullPhoto dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FullPhoto/Training12.ipynb_</ins> file for details.

### - Above90 / FaceOnly

1. **Training1:** First model training process. 
In this notebook file, a model based on VGG16 architecture were trained with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training01.ipynb_</ins> file for details.
2. **Training2:** Second model training process. 
In this notebook file, a model based on MobileNetV2 architecture were trained with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training02.ipynb_</ins> file for details.
3. **Training3:** Third model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, a model based on VGG16 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training03.ipynb_</ins> file for details.
4. **Training4:** Fourth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, a model based on MobileNetV2 architecture were trained with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training04.ipynb_</ins> file for details.
5. **Training5:** Fifth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning method with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training05.ipynb_</ins> file for details.
6. **Training6:** Sixth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning method with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training06.ipynb_</ins> file for details.
7. **Training7:** Seventh model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training07.ipynb_</ins> file for details.
8. **Training8:** Eighth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning method with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training08.ipynb_</ins> file for details.
9. **Training9:** Ninth model training process. 
In this notebook file, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training09.ipynb_</ins> file for details.
10. **Training10:** Tenth model training process. 
In this notebook file, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with the Above90/FaceOnly dataset.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training10.ipynb_</ins> file for details.
11. **Training11:** Eleventh model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, pre-trained VGG16 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training11.ipynb_</ins> file for details.
12. **Training12:** Twelfth model training process. 
In this notebook file, Data Augmentation operation were applied on Above90/FaceOnly dataset, pre-trained MobileNetV2 model were trained based on Transfer Learning and Fine-Tuning methods with this augmented data.
Accuracy and Loss charts were drawn for the Training and Validation data, and the results obtained by evaluating the trained model with the Test data were printed.
See <ins>_/Training/Above90/FaceOnly/Training12.ipynb_</ins> file for details.

---

# *<center>Tables</center>*

---

*While writing the accuracy values, the epochs whose Validation and Training Accuracy values were evaluated as the best were used throughout the training process.
Since these trainings are for observation purposes only, weights were not saved for the epochs evaluated as the best so the model was tested with the weights obtained in the last epoch.
Therefore, there is no Test Accuracy for the relevant weights when the epoch that is considered the best is not the last epoch.
In this case the Test Accuracy value was filled with a _-_ sign.*

### - ResizedData / FullPhoto

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.085 | 0.095 | 0.097 |
| Training2 | MobileNetV2 / Random  Weights | 0.866 | 0.198 | - |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.084 | 0.095 | 0.097 |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.957 | 0.302 | - |
| Training5 | VGG16 / ImageNet | 0.617 | 0.151 | - |
| Training6 | MobileNetV2 / ImageNet | 0.996 | 0.421 | 0.376 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.231 | 0.219 | 0.195 |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.872 | 0.370 | 0.334 |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.085 | 0.095 | 0.097 |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.696 | - |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.084 | 0.095 | 0.097 |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.995 | 0.662 | 0.574 |

### - ResizedData / FaceOnly

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.085 | 0.095 | 0.097 |
| Training2 | MobileNetV2 / Random  Weights | 0.835 | 0.173 | - |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.087 | 0.095 | 0.097 |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.929 | 0.527 | - |
| Training5 | VGG16 / ImageNet | 0.597 | 0.214 | 0.198 |
| Training6 | MobileNetV2 / ImageNet | 0.997 | 0.451 | 0.436 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.179 | 0.211 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.824 | 0.405 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.085 | 0.095 | 0.097 |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.728 | - |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.085 | 0.095 | 0.097 |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.996 | 0.703 | 0.650 |

### - FirstQuarter / FullPhoto

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.007 | 0.018 | - |
| Training2 | MobileNetV2 / Random  Weights | 1.000 | 0.018 | - |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.011 | 0.018 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.961 | 0.055 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.218 | 0.211 |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.309 | 0.404 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.833 | 0.255 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.999 | 0.418 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.007 | 0.018 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.255 | 0.439 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.011 | 0.018 | - |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 1.000 | 0.564 | - |

### - FirstQuarter / FaceOnly

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.007 | 0.018 | - |
| Training2 | MobileNetV2 / Random  Weights | 0.993 | 0.18 | 0.000 |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.008 | 0.018 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.096 | 0.055 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.291 | - |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.346 | 0.368 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.921 | 0.309 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.995 | 0.364 | 0.404 |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.009 | 0.018 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.273 | 0.263 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.009 | 0.018 | - |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 1.000 | 0.346 | 0.421 |

### - ThirdQuarter / FullPhoto

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.017 | 0.041 | - |
| Training2 | MobileNetV2 / Random  Weights | 1.000 | 0.031 | 0.000 |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.021 | 0.031 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.935 | 0.093 | 0.099 |
| Training5 | VGG16 / ImageNet | 1.000 | 0.572 | - |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.563 | 0.395 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.965 | 0.479 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.994 | 0.552 | 0.333 |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.012 | 0.031 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.667 | 0.469 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.021 | 0.020 | 0.025 |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 1.000 | 0.718 | 0.556 |

### - ThirdQuarter / FaceOnly

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.022 | 0.031 | - |
| Training2 | MobileNetV2 / Random  Weights | 0.980 | 0.021 | 0.012 |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.021 | 0.031 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.956 | 0.063 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.625 | 0.531 |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.562 | - |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.930 | 0.458 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.967 | 0.573 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.027 | 0.031 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.417 | 0.395 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.025 | 0.021 | 0.025 |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.997 | 0.583 | - |

### - Between80And90 / FullPhoto

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.030 | 0.041 | 0.017 |
| Training2 | MobileNetV2 / Random  Weights | 0.981 | 0.041 | 0.008 |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.031 | 0.041 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.933 | 0.214 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.537 | 0.400 |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.537 | - |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.950 | 0.479 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.973 | 0.579 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.028 | 0.041 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.0703 | 0.683 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.034 | 0.041 | - |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.999 | 0.686 | - |

### - Between80And90 / FaceOnly

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.302 | 0.025 | 0.033 |
| Training2 | MobileNetV2 / Random  Weights | 0.958 | 0.017 | 0.025 |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.036 | 0.025 | - |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.953 | 0.265 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.710 | 0.650 |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.579 | 0.608 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.926 | 0.603 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.935 | 0.620 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.030 | 0.041 | 0.017 |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.744 | 0.700 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.029 | 0.050 | - |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.999 | 0.694 | 0.625 |

### - Above90 / FullPhoto

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.024 | 0.024 | 0.024 |
| Training2 | MobileNetV2 / Random  Weights | 1.000 | 0.024 | - |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.021 | 0.024 | 0.024 |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.861 | 0.194 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.476 | - |
| Training6 | MobileNetV2 / ImageNet | 1.000 | 0.508 | - |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.943 | 0.516 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.921 | 0.468 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.014 | 0.024 | 0.024 |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.823 | 0.786 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.028 | 0.024 | - |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.997 | 0.742 | - |

### - Above90 / FaceOnly

| Name | Details | Training Accuracy | Validation Accuracy | Test Accuracy |
| :-----: | :-------: | :-----------------: | :-------------------: | :-------------: |
| Training1 | VGG16 / Random  Weights | 0.024 | 0.024 | - |
| Training2 | MobileNetV2 / Random  Weights | 0.993 | 0.024 | - |
| Training3 | VGG16 / Random  Weights / Data Augmentation | 0.026 | 0.024 | 0.024 |
| Training4 | MobileNetV2 / Random  Weights / Data Augmentation | 0.952 | 0.097 | - |
| Training5 | VGG16 / ImageNet | 1.000 | 0.750 | 0.690 |
| Training6 | MobileNetV2 / ImageNet | 0.997 | 0.645 | 0.667 |
| Training7 | VGG16 / ImageNet / Data Augmentation | 0.954 | 0.629 | - |
| Training8 | MobileNetV2 / ImageNet / Data Augmentation | 0.872 | 0.661 | - |
| Training9 | VGG16 / ImageNet / Fine-Tuning | 0.024 | 0.024 | - |
| Training10 | MobileNetV2 / ImageNet / Fine-Tuning | 1.000 | 0.734 | 0.730 |
| Training11 | VGG16 / ImageNet / Fine-Tuning / Data Augmentation | 0.023 | 0.024 | 0.024 |
| Training12 | MobileNetV2 / ImageNet / Fine-Tuning / Data Augmentation | 0.998 | 0.686 | - |

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