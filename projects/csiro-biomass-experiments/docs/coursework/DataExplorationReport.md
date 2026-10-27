> Historical IMSA coursework write-up. Descriptions, dataset counts, metrics, and leaderboard statements are preserved from the original and were not independently verified or reproduced for publication.

# Data statistics
There are 1785 different train samples, with metadata stored in the train.csv and images stored in the train folder. The test data only gives the image file and the target, it does not have any of the metadata found in the train.csv file.

# Data quality issues
There seems to be no missing values in the rows and columns of the train data. The dataset is also already balanced among its classes.

# Visualizations
The images are all large top down 2000x1000 RGB images of plants. 

# Initial preprocessing decisions
For the data I am using only the image data combined with the target choice and value for training. So all the metadata in the train.csv file will not be used. I also initialize everything into the propper files. I plan on creating one model with 5 outputs correlating to the 5 target values for this competition. I also set up image preprocessing, where I am resizing images to 224x224, scaling down by 255, and normalization. The training data is also augmented with flips, rotations, and changes in brightness and jitter. The train and validation data set have an image as input and the 5 targets as outputs, a 80/20 split is used between the train and validation.