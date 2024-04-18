# NeuralGuidance

This is the code base for *Leveraging the Human Ventral Visual Stream to Improve Neural Network Robustness* 
(Shao, Z., Ma, L., Li, B., & Beck, D. M., 2024, in prep)

:tada: See the abstract of our oral presentation [@VSS2024](https://www.visionsciences.org/talk-session/?id=164)

[//]: # (### figure)

[//]: # (### Abstract)

### Data:
- Original human neural data are retrieved from the Natural Scene Dataset (NSD) (Allen et al., 2022) publically available on [here](https://naturalscenesdataset.org/).

- Further processing of neural data for neural predictor training uses scripts in [neural_data_proc/](./neural_data_proc)

- Images used to train neural predictors are a selection from MSCOCO used in NSD for participants to view. Images used in neural-guidance training were obtained from [ImageNet](https://www.image-net.org/download.php) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). Details on these image sets are in the [*Methods*]() section of our manuscript.

[//]: # (- Fully trained weights of neural predictors used in our experiment have also been made available [here]&#40;https://osf.io&#41;. )


***
### Folder walkthrough

[**neural_data_proc/**](./neural_data_proc): 
All processing scripts needed to further clean and extract neural data from the NSD dataset.
Note: need [**AFNI**](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html) (Version AFNI_20.0.4) to run the scripts.

[**neural_predictor_training/**](./neural_predictor_training): 
Scripts used to train neural predictors. Need to have the fully processed neural data and images ready. 
Neural predictors used for ImageNet images and CIFAR-100 images have different structures to accomodate image difference. 
main_regular.py is for the regular neural predictor, and main_cifar.py is to run the version designed for CIFAR-100.

[**neural_guidance_training/**](./neural_guidance_training):
Scripts used to train the double-headed Resnet-18-based DNN to perform both classification and neural representation learning. 
Need to have fully trained neural predictors ready. Again, we have two versions to deal with imageNet and CIFAR-100. 
We included 7 neurally-guided models, 4 baseline models, and additional 5 WD-models (with different levels of weight decay
values that creates models with comparable level of output surface smoothness).

[**captioning/**](./captioning):
This is to adapt the neurally-guided DNNs to serve as the encoder for an image captioning task. 
The encoder-decoder structure and the pretrained-weights of decoders were obtained from [Show-Attend-Tell](https://arxiv.org/abs/1502.03044).

[**analysis/**](./analysis):  
- attacks: this folder contains scripts to run all adversarial attacks we investigated.
- texture_shape: this folder contains how we generated the texture-shape blended imageset and the evaluation of model's texture versus shape bias. 
- smoothness: this folder contains both the smoothness quantification and loss landscape (w.r.t input images) surface visualization. 
- rsa: this folder contains how the representation space similarity matrix was generated across all models used, along with the MDS visualization.
- noise_ceiling.py: this shows how to estimate the noise ceiling of neural data from each ROI using methods presented in the NSD.


