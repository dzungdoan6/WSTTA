
# About
This is the source code of the paper Weakly Supervised Test-Time Domain Adaptation for Object Detection

# Installation
Our installation is based on [Detectron2's installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
* Clone this repo, suppose the source code is saved in `[ROOT_DIR]/WSTTA`

* Install python >= 3.7 (tested on python 3.9)

* Install pytorch >= 1.8 and torchvision that matches the pytorch installation (tested on pytorch 2.0.1, torchvision 0.15.2, and pytorch-cuda 11.7)

* Install OpenCV
    ```
    pip install opencv-python
    ``` 
    (tested on opencv-python-4.8.0.76)

* Install the source code
    ```
    cd [ROOT_DIR]
    python -m pip install -e WSTTA
    cd [ROOT_DIR]/WSTTA
    ```

# Dataset Preparation

* Download Clear (Original) [KITTI dataset](https://www.cvlibs.net/datasets/kitti/).

* Download [KITTI-Fog](https://team.inria.fr/rits/computer-vision/weather-augment/).

* Place all clear images to `[ROOT_DIR]/WSTTA/data/kitti/clear` and all foggy images to `[ROOT_DIR]/WSTTA/data/kitti/fog`.

* Make sure `kitti_clear_train.json`, `kitti_clear_test.json`, `kitti_foggy_train.json`, and `kitti_foggy_test.json` can be found in `[ROOT_DIR]/WSTTA/data/kitti`.

# Running Experiments

* Train an object detector on the training set of KITTI

```
python WSTTA/train_detector.py --config-file configs/KITTI/kitti_faster_rcnn.yaml --imgs-dir data/kitti/clear --annos-file data/kitti/kitti_clear_train.json
```
After the training is finished, the source pretrained detector can be found in `[ROOT_DIR]/WSTTA/work_dir/kitti/clear`

* Adapt the source pretrained detector on the testing set of KITTI-Fog
```
python WSTTA/wstta_main.py --config-file configs/KITTI/kitti_faster_rcnn_wstta.yaml --imgs-dir data/kitti/fog --annos-file data/kitti/kitti_foggy_test.json --num-adapt 100 --mom-init 0.1 --mom-lb 0.005 --omega 0.99 --alpha 0.1 --psd-thr 0.8
```
After the WSTTA is finished, we should obtain the results of WSTTA shown in the Table 2a

* To obtain the results of Source shown in the Table 2a
```
python WSTTA/wstta_main.py --config-file configs/KITTI/kitti_faster_rcnn_wstta.yaml --imgs-dir data/kitti/fog --annos-file data/kitti/kitti_foggy_test.json --num-adapt 0
```



