# RecommendSystem
Based on collabrative filtering, this project predicts the ratings for every <userId> and <productId> combination in video_small_testing_num.csv.
  
The video_small_testing_num.csv datasets are a subset of the video_small_num.csv, each having the same columns as its parent.

## UserBasedCF
Using pearson similarity to calulate the similarity between the users. And using the most similar users' data to predict every <userId> and <productId> combination.
  
 
## ItemBasedCF
Using pearson similarity to calulate the similarity between the movies, using the most similar movies' data to predict every <userId> and <productId> combination. And by intergrating LSH, we filter part of movies that are not relevant to the movie we need to predict.
