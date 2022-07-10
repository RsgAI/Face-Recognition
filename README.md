
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

- _Python Version: **3.7.13**_
- _Numpy Version: **1.21.5**_
- _OpenCV(cv2) Version: **3.4.2**_
- _Pandas Version: **1.3.5**_

---

# *<center>Notebooks</center>*

---

## *Data Preparation*

### - Preparation

1. **Preparation1:** First Step of data preparation process. Reading raw images with opencv library, 
dataset is cleanning from non-quality redundancy and shrinking.
Dataset is splitting into Training, validation and Test data and ids are reset.
Performing the splitting process in the first stage will allow all models to be trained with the same Training data.
Thus, the performance change caused by the Training data difference will be prevented.
In this way, it will be possible to determine more accurately which model and parameters are better.
Data is saved as md5 files for future use.
See <ins>_/DataPreparation/Preparation1.ipynb_</ins> file for details.

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