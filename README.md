
# About
This is the source code of the paper [Weakly Supervised Test-Time Domain Adaptation for Object Detection](https://arxiv.org/abs/2407.05607)

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
# KITTI to KITTI-Fog
### Dataset preparation

* Download KITTI [here](https://www.cvlibs.net/datasets/kitti/).

* Download KITTI-Fog [here](https://team.inria.fr/rits/computer-vision/weather-augment/) or download our processed foggy images [here](https://drive.google.com/file/d/1zEM08vF8a2tMtNBSqj5WhFhbvQt4rsu3/view?usp=sharing).

* Place all clear images to `[ROOT_DIR]/WSTTA/data/kitti/clear` and all foggy images to `[ROOT_DIR]/WSTTA/data/kitti/fog`.

* Make sure `kitti_clear_train.json`, `kitti_clear_test.json`, `kitti_foggy_train.json`, and `kitti_foggy_test.json` can be found in `[ROOT_DIR]/WSTTA/data/kitti`.

### Running experiments

* Train an object detector on the training set of KITTI

```
python WSTTA/train_detector.py --config-file configs/KITTI/kitti_faster_rcnn.yaml --imgs-dir data/kitti/clear --annos-file data/kitti/kitti_clear_train.json
```
After the training is finished, the source pretrained detector can be found in `[ROOT_DIR]/WSTTA/work_dir/kitti/clear`

* Adapt the source pretrained detector on the testing set of KITTI-Fog
```
python WSTTA/wstta_main.py --config-file configs/KITTI/kitti_faster_rcnn_wstta.yaml --imgs-dir data/kitti/fog --annos-file data/kitti/kitti_foggy_test.json --num-adapt 100 --mom-init 0.1 --mom-lb 0.005 --omega 0.99 --alpha 0.1 --psd-thr 0.8
```
After the WSTTA is finished, we should obtain the results of `WSTTA` shown in the Table 2a

* To obtain the results of `Source` shown in the Table 2a
```
python WSTTA/wstta_main.py --config-file configs/KITTI/kitti_faster_rcnn_wstta.yaml --imgs-dir data/kitti/fog --annos-file data/kitti/kitti_foggy_test.json --num-adapt 0
```

# MSA-SYNTH Visible to Infrared

### Dataset preparation
* Download MSA-SYNTH [here](https://drive.google.com/file/d/1Db_zzwYvhdPDJbinAAzm8qjZlMb72OGT/view)

* Extract the MSA-SYNTH dataset to `[ROOT_DIR]/WSTTA/data/`

* Make sure `MSA-SYNTH/RGB/`, `MSA-SYNTH/IR/`, `MSA-SYNTH/rgb_train_cocostyle.json`, `MSA-SYNTH/rgb_test_cocostyle.json`, `MSA-SYNTH/ir_train_cocostyle.json`, and `MSA-SYNTH/ir_test_cocostyle.json` can be found in `[ROOT_DIR]/WSTTA/data/`

### Running experiments

* Train an object detector on the training set of Visible images

```
python WSTTA/train_detector.py --config-file configs/MSA-SYNTH/msa_synth_faster_rcnn.yaml --imgs-dir data/MSA-SYNTH/RGB --annos-file data/MSA-SYNTH/rgb_train_cocostyle.json
```

* Adapt the source pretrained detector on the testing set of Infrared images
```
python WSTTA/wstta_main.py --config-file configs/MSA-SYNTH/msa_synth_faster_rcnn_wstta.yaml --imgs-dir data/MSA-SYNTH/IR --annos-file data/MSA-SYNTH/ir_test_cocostyle.json --num-adapt 100 --mom-init 0.1 --mom-lb 0.005 --omega 0.94 --alpha 0.1 --psd-thr 0.8
```

