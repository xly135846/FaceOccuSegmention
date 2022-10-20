# FaceOccuSegmention

## Model
* Unet/DeepLabV3
* decode:MobileNetV2
* encode:upsampleing pooling
* SE-block and other 

## Loss
* cls
* dice_coef_loss
* tversky_loss
* focal_tversky
* tversky_cls
* boundary_loss
* dice_cls
* dice_ohem_cls

## Metic
* iou 
* acc
* dice_coef

## Once Time
* Snapdragon 845
* MobileNetV2
| image-size| MobilenetV2-size| CPU-xnnpack/ms| GPU/ms |
| :-----| ----: | :----: | :----: |
| 128 |  0.35 | 6.4 | 4.5 |
| 128 |  0.5 | 7.6 | 4.8 |
| 128 |  0.75 | 11.9 | 4.6 |
| 192 |  0.35 | 14.8 | 5.1 |
| 192 |  0.5 | 17.9 | 5.1 |
| 192 |  0.75 | 27.7 | 6.2 |
