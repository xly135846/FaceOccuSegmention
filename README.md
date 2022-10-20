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
* 骁龙845

| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
