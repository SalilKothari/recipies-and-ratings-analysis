# Recipes and Ratings Analysis 👨‍🍳
Analyzing Recipies and Ratings Dataset

# Introduction

Our dataset consists of information on recipes, as well as associated reviews and ratings for each of those recipes. Some important data regarding the recipes dataset to keep note of throughout the report are the nutrition values (calories, total fat, carbs, sugar, etc.), the average ratings, and the cooking time. 

We wanted to center our analysis around variables like the average rating, calories, and the cooking time, so we asked the question: what is the relationship between average rating and calories? 

Although our analysis is centered around this question, we still chose to explore all variables of interest, including cooking time and the other macronutrients. 

Lastly, we will give an overview of the dataset for convenience throughout the report:

* 234,429 rows
* Column Names of Interest
    * Average Rating: 'avg_rating'
        * Represents average rating of each recipe (since the dataset has a review for each recipe)
    * Cooking Time: 'minutes'
        * Represents cooking time in minutes
    * Calories: 'calories (#)'
        * Represents calories of a recipe
    * Carbohydrates: 'carbohydrates (PDV)'
        * Represents carbohydrates in a recipe, measured in percent daily value
    * Total Fat: 'total fat (PDV)'
        * Represents total fat in a recipe, measured in percent daily value
    * Sugar: 'sugar (PDV)'
        * Represents sugar in a recipe, measured in percent daily value
    * Protein 'protein (PDV)'
        * Represents protein in a recipe, measured in percent daily value


# Data Cleaning and Exploratory Analysis

Preview of Cleaned Recipes DataFrame:
| name                               |     id | minutes | contributor_id | submitted           | tags                 | n_steps | steps           | description       | ingredients             | n_ingredients |   user_id | recipe_id | date                | rating | review            | avg_rating | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | tags_length |
|:-----------------------------------|-------:|---------:|---------------:|:--------------------|:----------------------|--------:|:----------------|:------------------|:------------------------|--------------:|----------:|----------:|:--------------------|-------:|:-----------------|-----------:|-------------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|------------:|
| 1 brownies in the world best ever  | 333281 |      40 |         985201 | 2008-10-27 00:00:00 | ["'60-minutes-or-less'", ...] |      10 | ['heat the...', ...] | these are the...  | ['bittersweet...', ...] |             9 |    386585 |   333281 | 2008-11-19 00:00:00 |      4 | These were pretty... |          4 |        138.4 |              10 |          50 |           3 |             3 |                  19 |                   6 |          14 |
| 1 in canada chocolate chip cookies | 453467 |      45 |       1848091  | 2011-04-11 00:00:00 | ["'60-minutes-or-less'", ...] |      12 | ['pre-heat...', ...] | this is the...    | ['white sugar', ...]    |            11 |    424680 |   453467 | 2012-01-26 00:00:00 |      5 | Originally I was... |          5 |        595.1 |              46 |         211 |          22 |            13 |                  51 |                  26 |           9 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |     29782 |   306168 | 2008-12-31 00:00:00 |      5 | This was one...    |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |  1196280 |   306168 | 2009-04-13 00:00:00 |      5 | I made this for... |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |   768828 |   306168 | 2013-08-02 00:00:00 |      5 | Loved this. Be... |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |



Univariate Cooking Time Distribution:
<iframe
  src="assets/univariate-cooking-time-dist.html"
  width="400"
  height="300"
  frameborder="0"
></iframe>


Univariate Calorie (#) Distribution:
<iframe
  src="assets/univariate-calorie-dist.html"
  width="400"
  height="300"
  frameborder="0"
></iframe>


Univariate Average Rating (1-5) Distribution:
<iframe
  src="assets/univariate-rating-dist.html"
  width="400"
  height="300"
  frameborder="0"
></iframe>

Bivariate Cooktime vs Average Rating:
<iframe
  src="assets/bivariate-cooktime-rating-avg.html"
  width="400"
  height="300"
  frameborder="0"
></iframe>

Bivariate Total Fat vs Calories (#):
<iframe
  src="assets/bivariate-totalfat-calories.html"
  width="400"
  height="300"
  frameborder="0"
></iframe>



# Framing a Prediction Problem


We chose to predict the number of calories of a certain recipe since we were able to find strong relationships between the other macronutrients and calories. This is a regression problem since we are predicting a single value, and not classifying it into a group. Although it is not directly related to the question we posed for analysis, we still performed lots of analysis on the calories of a recipe and will focus mostly on calories for the remainder of the project.

We will initially be using a multiple linear regression model with 3 features: total fat, carbohydrates, and sugar. To evaluate the strength of this model, we will calculate the $R^2$ coefficient, which represents how strongly correlated the relationship between the predictor variables and our response variable is, thus representing the strength of the model.


# Baseline Model

Although we know how strong our model is on this data, we still need to see how strong it will be on generalized unseen data. Therefore, we will be conducting a training and test split to have the test data evaluate  our model instead of solely using the R^2 value.

Below, we are still using the same baseline model with the same features (total fat, carbs, and sugar), but we will split the data into training and testing data so that we can see how well our model would generalize to unseen data. Specifically, we will train our model on the training data, and evaluate its performance on both the training data and testing data to see if there is large variance between the 2. 



# Final Model

