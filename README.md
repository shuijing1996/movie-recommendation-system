# movie-recommendation-system
This repo aims to recommend new movies based on the provided movie name. The recommended movies and the provided movie should be in data/Sr._Data_Scientist_Assessment_Dataset.csv (this dataset serves as a movie database). The repo uses SentenceTransformer to embed Overview and Genre for each movie. The final vector combines sentence embedding with standardized Popularity, Vote_Count and Vote_Average. Finally, cosine similarity is used to calculate similarity between final vector of the provided movie and final vectors of the rest of the movies in the data set. The top n (user define) smallest cosine similarity movies will be provided back as recommendation. 

<br>

## Start the virtual environment

In order to run this repo, you need to install poetry first. Please follow the instructions in this [link](https://python-poetry.org/docs/) to install poetry first. 

1. To install required all required packages and dependencies

`poetry install`

2. To start the virtual environment

`poetry shell`

3. Install urllib3 version 1.26.6 for SentenceTransformer

`pip install urllib3==1.26.6`

<br>

## Get movies embedding

This script will get the final vector for each movie in the data set and store it in the data folder.

`ipython src/train/train.py`

<br>

## Get movie recommendation

This script will provide movie recommendations based on the provided movie. Since there are movies with the same names, in order to give precise recommendations, Release Date will be provided along with the movie names.

The require input **--target_movie**. The optional input **--recommend_count** (default is 5). 

<br>

### Example 1: Get 5 movie recommendations for "Puss in Boots: The Last Wish"

`ipython src/predict/predict.py -- --target_movie "Puss in Boots: The Last Wish" --recommend_count 5`

output:

*Given provided movie: Puss in Boots: The Last Wish_2022-08-24 (Release Date: 2022-08-24)*

*5 recommended movies are:*

*Puss in Book: Trapped in an Epic Tale (Release Date: 2017-06-16)*

*Puss in Boots: The Three Diablos (Release Date: 2012-02-07)*

*Puss in Boots (Release Date: 2011-10-28)*

*Scared Shrekless (Release Date: 2010-10-28)*

*Tad the Lost Explorer and the Mummy's Curse (Release Date: 2022-08-26)*
<br>

### Example 2: Get 5 movie recommendations for "Everything Everywhere All at Once"

`ipython src/predict/predict.py -- --target_movie "Everything Everywhere All at Once" --recommend_count 5`

output:

*Given provided movie: Everything Everywhere All at Once_2022-03-24 (Release Date: 2022-03-24)*

*5 recommended movies are:*

*Slate (Release Date: 2020-10-24)*

*Doctor Strange in the Multiverse of Madness (Release Date: 2022-05-04)*

*I Love America (Release Date: 2022-03-11)*

*Infinite Storm (Release Date: 2022-03-25)*

*The Lost City (Release Date: 2022-03-24)*

<br>

### Example 3: Get 5 movie recommendations for "Black Widow"

Since there are two Black Widow movies (one realease in 2021 and other in 1987)  found in the data set, the script will return recommendations for both Black Widow movies. 

`ipython src/predict/predict.py -- --target_movie "Black Widow" --recommend_count 5`

output:

*Given provided movie: Black Widow_2021-07-07 (Release Date: 2021-07-07)*

*5 recommended movies are:*

*Captain America: The Winter Soldier (Release Date: 2014-03-20)*

*Captain Marvel (Release Date: 2019-03-06)*

*Birds of Prey (and the Fantabulous Emancipation of One Harley Quinn) (Release Date: 2020-02-05)*

*Iron Man 2 (Release Date: 2010-04-28)*

*Zack Snyder's Justice League (Release Date: 2021-03-18)*

<br>

*Given provided movie: Black Widow_1987-02-06 (Release Date: 1987-02-06)*

*5 recommended movies are:*

*Momentum (Release Date: 2015-10-15)*

*A Fall from Grace (Release Date: 2020-01-17)*

*Taking Lives (Release Date: 2004-03-19)*

*Kiss the Girls (Release Date: 1997-09-29)*

*Copshop (Release Date: 2021-09-09)*

<br>

## Future Work
After sanity checking the recommendations for the three examples listed above, it is safe to say the movie recommendation system is valid and helpful. However the current system has a few limitations and can be further improved with more time given. 

1. The current system does not use Release_Date and Original_Language variables to make recommendations. Adding these two variables to the current system might improve the current system.
2. The recommendation can be made only when the system finds an exact match in the dataset with the provided movie. In the future, it might be helpful to allow fuzzy matches for movie names. For example, if the user provides "black widow" or "BlackWidow", the system should be able to identify that the user means "Black Widow" and provide recommendation based on "Black Widow".
3. It is possible to create an ensemble movie recommendation system. For example, besides the current system (sentence_embedding + consine_similarity), we can create two more ststmes : sentence_embedding + knn and sentence_embedding + Euclidean_distance. The final recommendations can be picked by majority vote by three systems. 
4. Current system can only provide recommendations based on one target movie. However, in reality users usually have a movie history. It will be helpful to recommend the movie based on the user's historical movie watching records. (related source: https://arxiv.org/abs/1904.06690)
