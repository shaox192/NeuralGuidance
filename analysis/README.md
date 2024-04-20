# NeuralGuidance/analysis

Using **sub1** and **V1** as an example

**Needs**: fully trained model in the current directory

### Adversarial attacks
We performed five different adversarial attacks currently popular in the field.


### Texture-shape bias evaluation


### Surface smoothness evaluation


### Run RSA and MDS



### Calculate noise ceiling
Noise ceiling quantifies the neural data quality, essentially finds out how consistent a voxel's response is 
when the same stimulus is repeated. We used the same method from the NSD paper. 
The noise ceiling of each voxel in a given ROI can be calculated using:
```bash
python noise_ceiling.py \
       --sub sub1 \
       --roi V1 \
       --beta-data ../neural_data_proc/sub1_betas \
       --aux-data ../neural_data_proc/sub1_nsd
```
This will generate a pickle file including the NC estimates for each voxel in V1, named "sub1_V1_NC.pkl" int the current directory
This pickle file can be used to generate NC box plot using [plot_noise_ceiling.ipynb](../plots/plot_noise_ceiling.ipynb)



