# NeuralGuidance/neural_data_proc/

Using **subject 1** and ROI **V1** as an example

### ROI extraction

1. Make sure AFNI (we used Version AFNI_20.0.4) is installed.  


2. Download data files: Kastner2015 ROI masks, nsdgeneral mask, and the anatomical volume from [NSD]() and stored in folder [](). 
   We provide the get_data_roi.sh to do this with command:
   ```bash
   bash ./get_data_roi.sh subj01 sub1
   ```
   **Note1**: The second argument is optional, but will change under what ID the subject's data is used in local directory. This will create a "sub1_nsd" folder.
   **Note2**: The experiment design file: nsd_expdesign.mat is also dowloaded. This is the same for all subjects.

3. use extract_roi.sh to extract ROI needed.
   Using V1 as an example:
   ```bash
   bash ./extract_roi.sh sub1 V1
   ```
   Choices of ROIs: V1, V2, V4, VO, PHC, LO, TO.

   You should see the mask files such as "sub1_V1.nii" in the sub1_nsd/ folder. Once repeating this procedure for all
   desirable ROIs, remember to visually check that your ROIs are in the right places using AFNI!

   **Note**: that most of the ROIs may have two sub-divisions. For example, there are V1v and V1d stored in the original Kastner atlas, corresponding to both the ventral division and dorsal division of V1. 
   One reason is that these halves, especially for earlier retinotopic regions, each responsible for half of the visual field 
   (e.g. upper and lower halves.). Therefore, we combine them to obtain the full ROI. This division of responsibility is more nuanced when we move up the hierarchy, reaching places like LO with two sub-divisions: LO1 and LO2.


4. Then use prep_roi.py, we can turn nifti formatted ROIs into numpy arrays for more convenient processing in the next sep. 
   Usage:
   ```bash
   python prep_roi.py --sub sub1 --roi V1
   ```
   Now you should see "sub1_V1.pkl" in sub1_nsd/ folder.


### Obtain and process functional data

1. The beta estimates files need to be obtained from NSD first 
   using: 
   ```bash
   python get_data_beta.py --sub-onl subj01 --sub-loc sub1 --num-ses 40
   ```
   
   **Note1**: Even data for one subject is huge!! 
   **Note2**: You may see an error message showing that only 38 session are downloaded. This is because two sessions of the NSD data have been withheld for the Algonauts challenge, although they will be released soon. 

2. Next we generate the test and val split of NSD data for a given ROI, e.g., V1. Neural responses have also been further PCA'ed for further cleaning. 
   Run prep_beta.py as follows: 
   ```bash
   python prep_beta.py --sub sub1 --roi V1
   ```
   The train/test split "sub1_V1_data.pkl" can be found in the newly created folder: sub1_data/



