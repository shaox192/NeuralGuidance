# NeuralGuidance/neural_predictor_training

Using **sub1** and **V1** as an example

## Training

Use the following to train neural predictor on one subject:
```bash
python train.py --sub sub1 \
                --roi V1 \
                --data-dir ../neural_data_proc/sub1_data \
                --save-dir ./ckpt \
                --lr 0.001 \
                --epk 40 \
                --save-interval 5
```
if --shuffle flag is added, the correspondence between image and corresponding neural data will be shuffled, 
serving as control condition to examine whether the neural predictors are effective.


