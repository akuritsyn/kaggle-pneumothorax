# [Kaggle SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

## Top 4% solution (50/1475) by [AlexeyK](https://www.kaggle.com/akuritsyn), [Wayfarer](https://www.kaggle.com/joven1997), and [Kudaibergen R](https://www.kaggle.com/kudaibergenu)

### The goal of the competition was to develop a model to classify (and if present, segment) pneumothorax (colapsed lungs) from a set of chest radiographic images. 

## The model is an ensemble of (1) Unet with EfficientNetB4 encoder on 512x512 images (in Keras) and (2) Unet with ResNet34 encoder on 1024x1024 images (Pytorch).

(1) Keras model was progressively trained from 256x256 to 512x512 size (due to limitations of Kaggle kernels upscaling to 1024x1024 was not feasible) 
- 256x256, trained from zero for 70 epochs, batch_size=16 (no grad. accum.), init_lr=1e-3
- 512x512, initialize by 256x256 model weights, trained for 16 epochs (to fit in 9hrs training time on Kaggle kernel), batch_size=4, grad_accum=4, , init_lr=1e-3
- 512x512, initialize by previous 512x512 model weights, trained for 16 epochs, batch_size=4, grad_accum=4, init_lr=1e-4
- 512x512, initialize by previous 512x512 model weights, trained for 16 epochs, batch_size=4, grad_accum=4, init_lr=3e-5

(2) Pytorch model was progressively trained from the 256x256 image size to the 512x512 image size.
