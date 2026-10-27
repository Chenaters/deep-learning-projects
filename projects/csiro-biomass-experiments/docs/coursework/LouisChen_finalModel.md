> Historical IMSA coursework write-up. Descriptions, dataset counts, metrics, and leaderboard statements are preserved from the original and were not independently verified or reproduced for publication.

### Final Model & Analysis

#### Final model
The final model I used was the DINO-based model. It used the vit_small_patch16_dinov3 as the pretrained backbone model and then extra MLP layers until the 5 outputs for the five biomass values of Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, and GDM_g.

#### Why this design
I chose the DINO model as backbone because this model is a massive self trained model, allowing it to be easily applied to this relatively smaller biomass dataset. Then the 5 regression outputs allows for the easy input of an image and the output of values the model determines from the image.

#### Results
The DINO model had a validation MAE of 6.610080, compared to the baseline ResNet50 model which had a validation MAE of 12.0844. So the DINO model performed far better.

#### Error analysis
The model can still make error when the topdown view of the grass is overlapping, causing some vegetation to not be seen. Being off by 6 grams per prediciton is also somewhat a lot, which means the model can still probably be improved.

#### Final Kaggle score and ranking
My final submission was given a score at around 0.50/1.00. The first place scores were are 0.65, and I placed roughly in the 2000th place.

#### Bias and fairness concerns
The model can have biases coming from the time of day the images were taken, and where it was taken. This bias can cause the model to not perform as well in different environments for predicting biomass from images.

#### Dataset limitations
The dataset is somewhat small with only around 300 images, so the model relies on these few images a lot.

#### Overfitting risks
Overfitting can happen with how small of a dataset is used. The augmentation done could have helped reduce it. And with the model validation results compared to train results, they seem to be somewhat similar.

#### Real-world deployment issues
Real world issues can come from different cameras, lighting conditions, and viewpoints of the biomass images. This can result in changes in model performance from these differences in image selection methods.