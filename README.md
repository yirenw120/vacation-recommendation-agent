# vacation-recommendation-agent

A machine-learning-driven vacation recommendation engine that predicts destination ratings and outputs personalized **Top-5** travel suggestions by combining user-level features (demographics, preferences, behavior) and destination-level attributes (climate zone, city type, average temperature, geo/environment).
## What this project does
This system learns relationships between user profiles and destination desirability by expanding historical ratings into a training set of *(user, destination)* pairs and fitting multiple predictive models. 

**Models included**
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- SVM
- K-means clustering
- LightGBM regression
- XGBoost regression  
These are combined using a **stacking meta-learner** (linear regression) trained on cross-validation predictions to produce a more stable final rating predictor than any single model.

## Recommendation scenarios
1) **No-user-info ranking:** returns Top-5 destinations based on aggregated historical ratings.
2) **Personalized Top-5:** predicts each destination’s rating using the stacked ensemble; optional filters can further refine results. 

## Data
Two input datasets:
- **User dataset**: individual characteristics + past trip ratings  
  `users = read.csv('src/Vacation_UsersData.csv')` 
- **Destination dataset**: attributes for each location  
  `dests = read.csv('src/Vacation_DestinationData.csv')`
  
Examples of user features include age, budget, trip duration preference, current latitude/longitude, and categorical preferences (climate, travel style, companion, etc.). 
Destination features include climate zone, city type, average temperature, latitude/longitude, etc.

## Requirements
This project uses the following R packages (among others):
```r
library(rpart)
library(randomForest)
library(e1071)
library(xgboost)
library(lightgbm)
