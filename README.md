# NeuralGuidance

Code base for *Leveraging the Human Ventral Visual Stream to Improve Neural Network Robustness* 
(Shao, Ma, Li, & Beck, 2024, in prep)

### figure

### Abstract






### Data:
- Original human neural data are retrieved from the Natural Scene Dataset (NSD) (Allen et al., 2022) publically available on [here](https://osf.io/yc97n/).

- Further fully processed data ready for neural predictor training have been made available [here](https://osf.io). Data are stored in __HDF5__ formats descriptions in ?. Scripts for processing steps to generate these data are in [neural_data_proc/](./neural_data_proc)

- Images used to train neural predictors are a selection from MSCOCO used in NSD for participants to view. The processed versions are included in the same osf repo [here](https://osf.io). Images used in neural-guidance training were obtained from [ImageNet](), [CIFAR-100](). Details on these image sets are in the [*Methods*]() section of our manuscript.

-Fully trained weights of neural predictors used in our experiment have also been made available [here](https://osf.io). 


***
### Folder walkthrough

#### neural_data_proc
All processing scripts needed to further clean and extract neural data from subject-1 of the NSD dataset. There is a mix of AFNI shell scripts to extract all the ROI masks needed and further processing scripts. 

therefore are __not directly runnable__ but serve as a demonstration

**Special dependencies**:
 1. AFNI: Version AFNI_20.0.4

#### neural_predictor_training
Scripts used to train neural predictors. Need to have the fully processed neural data and images ready. Neural predictors used for ImageNet images and CIFAR-100 images are different in the first convolution layer. main_regular.py is for the regular neural predictor, and main_cifar.py is to run the version with modification.

**Special dependencies**:
 1. PyTorch

#### neural_guidance_training
Scripts used to train double-headed Resnet-18-based DNN to perform both classification and neural representation learning. Need to have fully trained neural predictors ready. Again, we have two versions to deal with imageNet and CIFAR-100. We included five ROI models (with two additional shown in the Supporting Information) and four different baseline models to support our conclusions. See the manuscript for a description of each.

#### captioning
This is to adapt the DNNs trained with neural-guidance to serve as the encoder for an image captioning task. The decoders have pretrained-weights provided by [Show-Attend-Tell]()

#### analysis
-attacks: this folder contains scripts to run all five adversarial attacks we included in the paper, and the transfer attack.
-texture_shape: this folder contains how we generated the texture-shape blended imageset and the evaluation of model's texture versus shape bias. 
-smoothness: this folder contains both the smoothness quantification and loss landscape (w.r.t input images) surface visualization. 
-rsa: this folder contains how the representation space similarity matrix was generated across all models used in our work.

**Special dependencies**:
 1. FoolBox
 2. autoattack


