> Historical IMSA coursework write-up. Descriptions, dataset counts, metrics, and leaderboard statements are preserved from the original and were not independently verified or reproduced for publication.

## Proposal
**Kaggle competition link:**
- https://www.kaggle.com/competitions/csiro-biomass
###
**Problem description (in own words):**
- From information images of pastures, predict a target value from 5 key biomass target quantities in the ground.  
###
**Dataset summary:**
- There are 5 categories of quantities to be predicted: dry green vegetation, dry dead material, dry clover biomass, green dry mattter, and total dry biomass.
- The training images have the ground truth values for the 5 categories along with species present in pasture, sampling date, Australian state, average pasture height, and normalized difference vegatation index.
- 357 files to train from.
- 800 images in the testing set

###
**Initial baseline idea:**
- I could probably just use images and their target values for training and creating the model. The model would need an image input along with the target biomass value, and the output would need to be some numerical value. A multilayer CNN can work for this project. I could also create 5 models for this project and switch between them based on which target biomass value is needed.