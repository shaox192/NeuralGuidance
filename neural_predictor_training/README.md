# NeuralGuidance - neural_predictor_training

## ROI extraction

**Prerequisite**:
 1. Have AFNI (preferably Version AFNI_20.0.4) installed.

 1. Download data files: Kastner mask with the anatomical delineation of ROIs used in our work, nsdgeneral mask, and the anatomical volume from [NSD]() and stored in folder [](). We provide the get_data_roi.sh to do this with command: 

 2. use extract_roi.sh to extract ROI needed. for example, if we want to use V1:   . Note that most of the ROIs may have two sub-divisions. For example, there are V1v and V1d stored in the original Kastner atlas, corresponding to both the ventral division and dorsal division of V1. One reason is that these halves, especially for earlier retinotopic regions, each responsible for half of the visual field (e.g. upper and lower halves.). Therefore, we combine them to obtain the full ROI. This division of responsibility is more nuances when we move up the hierarchy, reaching places like LO with two sub-divisions: LO1 and LO2.

 3. Then use prep_roi.py, we can turn nifti formatted ROIs into numpy arrays for more convenient processing in the next sep. usage:

## Obtain neural data and further prepping

 1. To obtain neural data, the beta files need to be obtained first using: get_data_beta.sh. Note, even data for one subject is huge. You may see an error message showing that only 38 session are downloaded. This is because two sessions of the NSD data have been withheld for an [Algonauts challenge](), although they will be released soon. 

 2. Running prep_beta.py will generate a data __hdf5__ containing test and val split of correct correspondance between each MSCOCO image that the participant saw, and the final neural representation corresponding to each image. neural responses have also been further PCA'ed for further cleaning. Run prep_beta.py as follows: 
