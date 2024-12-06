
# Recipes and Ratings Analysis 👨‍🍳
Analyzing Recipes and Ratings Dataset

By: Abhi Attaluri & Salil Kothari

# Introduction

Our dataset consists of information on recipes, as well as associated reviews and ratings for each of those recipes. Some important data regarding the recipes dataset to keep note of throughout the report are the nutrition values (calories, total fat, carbs, sugar, etc.), the average ratings, and the cooking time. 

We wanted to center our analysis around variables like the average rating, calories, and the cooking time, so we asked the question: what is the relationship between average rating and calories? 

We feel this is an important question to explore because it is important to try and find relationships between variables which we would never think could be related with each other, and doing so with something that everyone can relate with, such as food and recipes, puts a fun twist to data analysis. 

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

## Data Cleaning Steps:

1. First, we were given 2 datasets - one containing only the recipe info, and the other containing the reviews info. In order to work with one dataset only, we merged the 2 using the recipe ID, which was common across the 2 datasets. This is why you may see the same recipe multiple times - each of them is associated with its own unique review for that recipe.
2. The next step we took was to fill all the ratings of '0' with null (NaN) values. We did this because we wanted to calculate the average rating per recipe, and many ratings of '0' were not actual ratings but rather just people leaving comments on recipes. Therefore, replacing them with NaN values makes it so that the average rating metric was accurate and not skewed by the multiple '0's.
3. Next, the original dataframe has a 'nutrition' column, which was a string of numbers representing each of the macronutrients for that recipe - this included calories, total fat, sugar, protein, and more. To make these easier to manipulate, we converted the 'nutrients' value into a list, and then added each of the macronutrients as their own columns. 
4. The final big step we did was changing the data types of a lot of the variables in the original dataframe. We did this so that these values could be manipulated easier later on if we chose to use them. The transformations we performed are as follows:
    * 'date' column: string -> pandas Datetime object
    * 'submitted' column: string -> pandas Datetime object
    * 'id' column: string -> int64 object
    * 'contributor_id' column: string -> int64 object
    * 'n_ingredients' column: float -> int64 object
    * 'tags' column: string -> list
        * 'tags_length' column: new column we added by taking length of 'tags' list


After performing all of these transformations and cleaning steps, we get the following dataframe: (some of the values with particularly long string are shortened with an ellipsis for viewing pleasure)

## Preview of Cleaned Recipes DataFrame:

```py
print(df_new.head().to_markdown(index=False))
```

| name                               |     id | minutes | contributor_id | submitted           | tags                 | n_steps | steps           | description       | ingredients             | n_ingredients |   user_id | recipe_id | date                | rating | review            | avg_rating | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | tags_length |
|:-----------------------------------|-------:|---------:|---------------:|:--------------------|:----------------------|--------:|:----------------|:------------------|:------------------------|--------------:|----------:|----------:|:--------------------|-------:|:-----------------|-----------:|-------------:|----------------:|------------:|-------------:|--------------:|--------------------:|--------------------:|------------:|
| 1 brownies in the world best ever  | 333281 |      40 |         985201 | 2008-10-27 00:00:00 | ["'60-minutes-or-less'", ...] |      10 | ['heat the...', ...] | these are the...  | ['bittersweet...', ...] |             9 |    386585 |   333281 | 2008-11-19 00:00:00 |      4 | These were pretty... |          4 |        138.4 |              10 |          50 |           3 |             3 |                  19 |                   6 |          14 |
| 1 in canada chocolate chip cookies | 453467 |      45 |       1848091  | 2011-04-11 00:00:00 | ["'60-minutes-or-less'", ...] |      12 | ['pre-heat...', ...] | this is the...    | ['white sugar', ...]    |            11 |    424680 |   453467 | 2012-01-26 00:00:00 |      5 | Originally I was... |          5 |        595.1 |              46 |         211 |          22 |            13 |                  51 |                  26 |           9 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |     29782 |   306168 | 2008-12-31 00:00:00 |      5 | This was one...    |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |  1196280 |   306168 | 2009-04-13 00:00:00 |      5 | I made this for... |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |
| 412 broccoli casserole             | 306168 |      40 |         50969  | 2008-05-30 00:00:00 | ["'60-minutes-or-less'", ...] |       6 | ['preheat oven...', ...] | since there are... | ['frozen broccoli...', ...] |            9 |   768828 |   306168 | 2013-08-02 00:00:00 |      5 | Loved this. Be... |          5 |        194.8 |              20 |           6 |          32 |            22 |                  36 |                   3 |          10 |


## Univariate Graphs

Below, we have 3 graphs which represent the distributions of variables of interest that we wanted to analyze further:


The first graph is a distribution of the cooking time in minutes. We noticed that the cooking time distribution was right-skewed due to some recipes having very large outliers, but it seemed that overall the cooking time was centered between a range of 20-45 minutes. This helped us towards answering our question because we were able to find out that the cooking time was right-skewed, and that may affect any analysis we do on its relationship with other variables. 

Univariate Cooking Time Distribution:
<iframe
  src="assets/univariate-cooking-time-dist.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>




The second graph is a distribution of the calories. We noticed that this graph was also relatively right-skewed due to large outliers, and the number of calories for most recipes in this dataset was around 150-500. This helped us figure out the shape of the calories distribution, which we figured could affect its relationship with the average rating because some outliers could affect any predictions we made later on.

Univariate Calorie (#) Distribution:
<iframe
  src="assets/univariate-calorie-dist.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>


The last graph is a distribution of the average rating of recipes in this dataset. The largest thing to notice is that this graph is extremely left-skewed, as almost all of the recipes are rated very highly, and there are only slight differences in the ratings. This graph told us the most about our dataset because we realized that comparing calories or cooking time to average rating might not show any conclusive results since there is so little variation in the average rating. 

In other words, we realized at this point in our analysis that variations in calories or cooking time might not have any effect on variation in the average rating, which we were able to prove in the next part of our analysis with the bivariate graphs.

Univariate Average Rating (1-5) Distribution:
<iframe
  src="assets/univariate-rating-dist.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>


As we alluded to above, this graph below is a scatterplot representing the relationship between the cooking time and the average rating of recipes in the dataset. The main trend to notice here is that the scatter is very random, and the only pattern we can see is that there are more points near the higher ratings between 4.5-5, and this is only because most of the recipes in this dataset have ratings between 4.5 and 5. 

Therefore, at this point in our analysis, we were able to conclude that exploring the relationship between cooking time and average rating, or calories and average rating, would not yield too much in terms of a real relationship. 

Bivariate Cooktime vs Average Rating:
<iframe
  src="assets/bivariate-cooktime-rating-avg.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>


Overall, the graph above helped us pivot our question moreso to exploring the relationship between different macronutrients, as we noticed that the distributions of those were less skewed and a lot more predictable. Additionally, it also makes intuitive sense that the macronutrients are related to each other, since all of them combined make up a recipe. Specifically, we pivoted to exploring the relationship between total fat (in PDV) and calories, which we will see below:

Bivariate Total Fat vs Calories (#):
<iframe
  src="assets/bivariate-totalfat-calories.html"
  width="900"
  height="400"
  frameborder="0"
></iframe>

So, as we can see from the graph above, there is a relatively strong positive relationship between the total fat and calories, which - as alluded to above - makes sense because total fat is one of the many macronutrients which make up the number of calories in a recipe.


So, overall, these graphs were a pivotal moment in our report because they made us shift our focus to analyzing the macronutrients because we realized that it makes more sense to focus a prediction problem around the numerical columns since we still wanted to focus on calories. 

## Grouping Relationships between Variables

To confirm our hypothesis and motivations for switching to the focus to macronutrients, we created a table that groups together many variables of interest, and then computes the correlation between them. We did this so we could also further understand what kinds of relationships exist between many of the numerical variables of interest in this dataset:

| Variable 1          | Variable 2          |   Correlation |
|:--------------------|:--------------------|--------------:|
| minutes             | avg_rating          |    0.00196393 |
| minutes             | n_steps             |    0.0116946  |
| calories (#)        | avg_rating          |    0.0135263  |
| n_steps             | calories (#)        |    0.152484   |
| total fat (PDV)     | sugar (PDV)         |    0.403276   |
| total fat (PDV)     | carbohydrates (PDV) |    0.459807   |
| total fat (PDV)     | protein (PDV)       |    0.510125   |
| protein (PDV)       | calories (#)        |    0.593704   |
| sugar (PDV)         | calories (#)        |    0.681099   |
| carbohydrates (PDV) | calories (#)        |    0.812777   |
| total fat (PDV)     | calories (#)        |    0.869702   |

The value in the 'correlation' column is the correlation coefficient, and it effectively represents the relationship between the 2 variables, where values closer to 1 represent a stronger positive relationship. 

As we hypothesized, macronutrients like sugar, carbohydrates, and total fat all have a relativelty strong positive relationship with the calories, which makes sense because those are foods that make up the calories of a recipe. 

## Imputation

To explore the possibility of imputation, we wanted to see the ratios of missing values (NaNs) in each of the columns to confirm if we needed to perform any more imputation besides our initial imputation of the average rating, as described above.

The table of missing values for each of the columns is as follows:

| Column Name         |   Percentage of NaN Values |
|:--------------------|---------------------------:|
| name                |                4.26568e-06 |
| id                  |                0           |
| minutes             |                0           |
| contributor_id      |                0           |
| submitted           |                0           |
| tags                |                0           |
| n_steps             |                0           |
| steps               |                0           |
| description         |                0.000486288 |
| ingredients         |                0           |
| n_ingredients       |                0           |
| user_id             |                4.26568e-06 |
| recipe_id           |                4.26568e-06 |
| date                |                4.26568e-06 |
| rating              |                0.0641388   |
| review              |                0.00024741  |
| avg_rating          |                0.0118458   |
| calories (#)        |                0           |
| total fat (PDV)     |                0           |
| sugar (PDV)         |                0           |
| sodium (PDV)        |                0           |
| protein (PDV)       |                0           |
| saturated fat (PDV) |                0           |
| carbohydrates (PDV) |                0           |
| tags_length         |                0           |

Looking at the table above, we can see that the only columns of relevance that have a significant amount of missing (NaN) values is the 'rating' and 'average rating'. However, we purposefully replaced all the ratings of 0 with NaN values because we did not want that value of 0 to affect the accuracy of the average rating, since many reviews with a rating of 0 were just people leaving comments, and not actual reviews.

Besides the average rating, we can observe that the rest of the relevant variables, which are those that we are exploring through our analysis, all have 0 missing values in their columns, which is why we chose to not do any further imputation besides the initial changes we made to the 'rating' column.


# Framing a Prediction Problem

Our problem is predicting the number of calories in a specific recipe since we were unable to find strong relationships between calories and other nutrients. Since we are predicting the single numeric value of calories, this is a regression problem.

The variable to predict is calories (#), since we were most curious about it.

The metric we are using to evaluate our model is Mean Squared Error, because it is best for a Linear Regression Model. Additionally, we examined $R^2$, the Pearson Correlation Coefficient, which showed us the strength of relationships between the target variable and various features and helped us understand the strength of our model. We chose this over other metrics because our baseline model was a multiple linear regression model with only 3 features, so looking at the MSE and $R^2$ was the best metric at that point in time. Additionally, using something like accuracy would not be applicable for this problem since we are looking at a numerical prediction, not a classification. 

At the time of prediction, all we knew were the macronutrient values, including total fat, sugar, and carbohydrates. We also had data about reviews and later on, engineered features such as the number of tags attached to a recipe.

We would go on to use fat, carbohydrates, and sugar in our Baseline Model.


# Baseline Model

We will initially be using a multiple linear regression model with 3 features: total fat, carbohydrates, and sugar. To evaluate the strength of this model, we will calculate both the MSE and the $R^2$ coefficient, which represents how strongly correlated the relationship between the predictor variables and our response variable is, thus representing the strength of the model. The total fat, sugar, and carbohydrates are all measured in PDV (percent daily value), so we did not feel the need to apply any standardization transformations to it. 

To use these variables, we employed the changes as described in section 2: Data Cleaning and Exploratory Analysis. Specifically, making each of these values into its own column allowed us to easily use them as features in our model as shown below:

```py
multi_model = LinearRegression()
multi_model.fit(X=df_new[['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)']], y=df_new['calories (#)'])
```

This code above fit our model on the 3 features, and identified the variable we were going to predict, which was calories. 

Below, we computed the mean squared error for the model, by using the mean_squared_error and .predict() functions, which are a part of sklearn. 

```py
mean_squared_error(df_new['calories (#)'], 
                   multi_model.predict(df_new[['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)']]))
```
>>> 8834.033177215146

Now, as for the $R^2$ coefficient, we can use the .score() method:

```py
multi_model.score(df_new[['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)']], df_new['calories (#)'])
```

>>> 0.9740288778674402

From these 2 calculations, we computed an MSE of **8834.03** and an $R^2$ value of **0.97**, which is quite strong, and confirms our hypothesis that macronutrients are a good predictor of the calories.

Although we know how strong our model is on this data, we still need to see how strong it will be on generalized unseen data. Therefore, we will be conducting a training and test split to have the test data evaluate  our model instead of solely using the $R^2$ value.

Below, we are still using the same baseline model with the same quantitative, features (total fat, carbs, and sugar), but we will split the data into training and testing data so that we can see how well our model would generalize to unseen data. Specifically, we will train our model on the training data, and evaluate its performance on both the training data and testing data to see if there is large variance between the 2. We will be using sklearn's train_test_split function to conduct this split:

```py
from sklearn.model_selection import train_test_split
```

```py
X = df_new[['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)']]
y = df_new['calories (#)']
# using default split of 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)
```

Now, the difference is that we fit the model on the training set only, and then use the model to predict values in the test set:

```py
multi_model_train = LinearRegression()
multi_model_train.fit(X_train, y_train)
```

First, we will check our model's performance on the training set:

```py
pred_train = multi_model_train.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
mse_train
```

>>> 9261.011748852132

From this computation, we got an MSE of **9261.01** for our training set.

Next, we will compare this value to the testing MSE:

```py
pred_test = multi_model_train.predict(X_test)
mse_test = mean_squared_error(y_test, pred_test)
mse_test
```

>>> 7564.291787226931

From this computation, we got an MSE of **7564.29**. 

We can see here that our testing MSE was much lower than the training MSE, which indicates that there may be some noise in our training data that is affecting the model's performance on the training data. From our analysis of the dataset in the earlier parts, we saw that there were lots of outliers in the some of the nutritional data values, which means our training data may have been affected by this. Another possibility is that our model may be underfitting, which means it is too simple and can be made more complex to improve the overall model performance.

So, from our results above, we can see that our current baseline model that uses 3 features (total fat, carbs, and sugar) to predict the calories of a recipe is relatively strong, but may be a bit too simple and can be improved by transforming current features or creating new ones to make the model more complex, potentially increasing its performance. We will explore how to do so in the next section. 

# Final Model

For our final model, we decided that one new feature we will add is choosing hyperparameters before fitting our model to the data so that we increase the model's complexity. Specifically, we chose Polynomial Degrees as our first hyperparameter since we can apply that to our current numerical features (total fat, carbs, and sugar). We will use k-fold cross validation to tune the polynomial degree in order to find the 'best' degree - where 'best' in this context refers to the degree that achieves the best model performance. Similar to what we did previously, we will compute the training and test errors to evaluate our model's performance. We chose to do this as one of the hyperparamters because we felt that there was a chance the relationship was not linear. Specifically, although we saw a relatively linear relationship for this data, we were not sure about how it would perform on unseen data, so we wanted to find the best Polynomial Degree for this data. Additionally, this is a very common hyperparameter that is used on numerical data, and it often improves the model's performance because we are tuning the hyperparameter before applying it to the final model, which is why it is ideal for this task. 

Secondly, we will one-hot encode the 'tags_length' variable by splitting it into bins, and one-hot encoding those bins. We are doing this because we hypothesized that the length of the tags may indicate a recipe's popularity, and more popular recipes may also correlate to higher-calorie recipes since those are the ones that people generally tend to enjoy more. Therefore, one-hot encoding this and incorporating it into our model will not only increase our model's complexity but also provide other data for the model to work with to better predict the calories, making it ideal for this task.

Additionally, we will be performing k-fold cross validation because we observed that our test MSE was much lower for our baseline model, which may indicate that it was overfitting to the test set. To prevent this, we will have 'k' validation sets, and compare the performance with those validaiton sets before evaluating our model on the test set.


So, as described above, we decided to one-hot encode the 'tags_length' variable. However, before doing so, we had to do some more data manipulation and split the 'tags_length' variable into bins, where each bin value represents what 5-interval range the tag length was in for that recipe. For instance, if the bin value was a value of 15, that means that the tag length for that recipe was between 10 and 15 (inclusive, exclusive). In order to perform this transformation, we used the pd.cut() function, which is often used for binning variables:

```py
bins = np.arange(0, 51, 5)
labels = bins[1:]
make_tags_length_bins = Pipeline([
    ('binning', FunctionTransformer(lambda df: pd.cut(
        df_new['tags_length'],
        bins=bins,
        labels=labels,
        right=False
    ).astype(int).to_frame(name='tags_length_bins')))
])

binned_values = make_tags_length_bins.fit_transform(df_new)
df_new = pd.concat([df_new, binned_values], axis=1)
df_new['tags_length_bins'] = df_new['tags_length_bins'].astype('category')
```

This ended up creating a new column called 'tags_length_bins', and we made it a category so that it could be used for One-Hot Encoding. 

Now, we will perform the same train-test split as before, but this time include our new feature 'tags_length_bins':

```py
X = df_new[['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)', 'tags_length_bins']]
y = df_new['calories (#)']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)
```

Next, after performing the split, this is where we actually build the pipeline, since we are incorporating 2 new features into our original Linear Regression model (note that we are still using the same baseline Linear Regression model):

```py
preprocessing = make_column_transformer(
    (OneHotEncoder(drop='first'), ['tags_length_bins']),
    (PolynomialFeatures(), ['total fat (PDV)', 'carbohydrates (PDV)', 'sugar (PDV)'])
)

hyperparams = {
    'columntransformer__polynomialfeatures__degree': (1, 16)
}

searcher = GridSearchCV(
    make_pipeline(preprocessing, LinearRegression()),
    param_grid=hyperparams,
    cv=5,
    scoring='neg_mean_squared_error',
    error_score='raise'
)
searcher
```

Since we are performing transformation on different columns, we needed to use sklearn's make_column_transformer() function, which allows us to perform transformations in our pipeline on specific columns. So, as you can see above, we performed the One-Hot Encoding for the binned tag lengths, and then incorporated the PolynomialFeatures() feature, testing degrees 1-15 for our macronutrients' coefficients. 

After performing the transformations, we were then ready to make the pipeline and perform k-fold cross validation, which is done automatically using GridSearchCV, another class in sklearn. The 'cv = 5' argument tells us that we have 5 folds, meaning there are 5 validations sets that we compare the training set to in order to improve the model's performance. The reason this improves the model's performance is because each data point is used for training 4 times and validation once, which allows us to average the performance (measured in MSE) and then apply it to the model, giving us confidence that our model can generalize pretty well to unseen data since we performed it more times than any regular base model. 

This technique helps find the best hyperparameter, and at the end of the process, we found the following results:

As suspected, the relationship is indeed linear, so running the .best_params method tells us the best hyperparameter for our model, which in our case was a polynomial degree of 1:

```py
searcher.best_params_
```
>>> {'columntransformer__polynomialfeatures__degree': 1}


So, now that we found what hyperparameter our model used to optimize its performance, we can actually see its performance for ourselves by employing the same technique we did above: comparing the training and testing MSEs:

```py
pred_train = searcher.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
mse_train
```
>>> 9235.80

```py
pred_test = searcher.predict(X_test)
mse_test = mean_squared_error(y_test, pred_test)
mse_test
```
>>> 7540.77

As we can see, the model seems to now perform well on generalized unseen data because it has a much lower test error than training error, and this is after we performed k-fold cross validation. Additionally, we can notice that the MSE values themselves are also smaller with this model, indicating that while the improvement was not much, one-hot encoding the tag lengths seemed to help improve the model's overall performance. Lastly, it seems that the relationship between the macronutrients (total fat, carbs, and sugar) that we used and the calories is linear, as our model performed best with a polynomial degree of 1.

Overall, since the MSE is much lower for our final model and the test error is much lower than the training error, that shows that our final model is a slight improvement from our base model. This improvement occurred primarily because we increased the complexity by adding the One-Hot Encoding of 'tags_length', and because we used k-fold cross validation to optimize our hyperparameter.


# Conclusion

So, at the end of all this analysis on the recipes, what did we learn? 

We were able to conclude that this dataset did not have too many categorical variables we could use for our model, and instead that the macronutrients were the most interesting the explore because they were very highly correlated with each other. Therefore, this allowed us to create a model using some of the macronutrients to predict calories, where we concluded that given total fat, sugar, and carbohydrates of a recipe we can pretty accurately predict the calories of that recipe. This also aligns with our initial hypothesis, and does make sense intuitively!