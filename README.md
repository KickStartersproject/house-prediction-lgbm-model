# house-prediction-lgbm-model

"""

## Description

Application provides an interface to select independent variable filters for predicting the prices of houses in the Iowa, US. Prediction for house prices was developed using the Kaggle House Prices - Advanced Regression Techniques competition dataset.

## Data

The dataset is available at [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Goal

To predict the price of a houses in Ames Iowa, US using certain selected features from the dataset.

## Features

The dataset contains the following features:

* **SalePrice**: the property's sale price in dollars. This is the target variable that you're trying to predict.
* **MSSubClass**: The building class
* **MSZoning**: The general zoning classification
* **LotFrontage**: Linear feet of street connected to property
* **LotArea**: Lot size in square feet
* **Street**: Type of road access
* **Alley**: Type of alley access
* **LotShape**: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
* RoofMatl: Roof material
* Exterior1st: Exterior covering on house
* Exterior2nd: Exterior covering on house (if more than one material)
* MasVnrType: Masonry veneer type
* MasVnrArea: Masonry veneer area in square feet
* ExterQual: Exterior material quality
* ExterCond: Present condition of the material on the exterior
* Foundation: Type of foundation
* BsmtQual: Height of the basement
* BsmtCond: General condition of the basement
* BsmtExposure: Walkout or garden level basement walls
* BsmtFinType1: Quality of basement finished area
* BsmtFinSF1: Type 1 finished square feet
* BsmtFinType2: Quality of second finished area (if present)
* BsmtFinSF2: Type 2 finished square feet
* BsmtUnfSF: Unfinished square feet of basement area
* TotalBsmtSF: Total square feet of basement area
* Heating: Type of heating
* HeatingQC: Heating quality and condition
* CentralAir: Central air conditioning
* Electrical: Electrical system
* 1stFlrSF: First Floor square feet
* 2ndFlrSF: Second floor square feet
* LowQualFinSF: Low quality finished square feet (all floors)
* GrLivArea: Above grade (ground) living area square feet
* BsmtFullBath: Basement full bathrooms
* BsmtHalfBath: Basement half bathrooms
* FullBath: Full bathrooms above grade
* HalfBath: Half baths above grade
* Bedroom: Number of bedrooms above basement level
* Kitchen: Number of kitchens
* KitchenQual: Kitchen quality
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
* Functional: Home functionality rating
* Fireplaces: Number of fireplaces
* FireplaceQu: Fireplace quality
* GarageType: Garage location
* GarageYrBlt: Year garage was built
* GarageFinish: Interior finish of the garage
* GarageCars: Size of garage in car capacity
* GarageArea: Size of garage in square feet
* GarageQual: Garage quality
* GarageCond: Garage condition
* PavedDrive: Paved driveway
* WoodDeckSF: Wood deck area in square feet
* OpenPorchSF: Open porch area in square feet
* EnclosedPorch: Enclosed porch area in square feet
* 3SsnPorch: Three season porch area in square feet
* ScreenPorch: Screen porch area in square feet
* PoolArea: Pool area in square feet
* PoolQC: Pool quality
* Fence: Fence quality
* MiscFeature: Miscellaneous feature not covered in other categories
* MiscVal: $Value of miscellaneous feature
* MoSold: Month Sold
* YrSold: Year Sold
* SaleType: Type of sale
* SaleCondition: Condition of sale

## Usage

```bash
# clone the repo
git clone https://github.com/KickStartersproject/house-prediction-lgbm-model.git

# change to the repo directory
cd house-prediction-lgbm-ap

# create a virtualenv
py -m venv venv

# activate virtualenv for linux or mac
source venv/bin/activate

# activate virtualenv for windows
# venv\Scripts\activate

# install dependencies (this would install flask and other python libraries)
pip install -r dependencies.txt

# open app in visual studio code and on terminal run the following command to set flask main app
set FLASK_APP=main.py

```

## Model Development

### Model

The model is based on a [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) algorithm.

### Training

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(max_depth=3, 
                    n_estimators = 100, 
                    learning_rate = 0.2,
                    min_child_samples = 10)
model.fit(x_train, y_train)
```

Grid Search Cross Validation is used for hyper parameters of the model.

```python
from sklearn.model_selection import GridSearchCV

params = [{"max_depth":[3, 5], 
            "n_estimators" : [50, 100], 
            "learning_rate" : [0.1, 0.2],
            "min_child_samples" : [20, 10]}]

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      cv=5)

gs_knn.fit(x_train, y_train)
gs_knn.score(x_train, y_train)

pred_y_train = model.predict(x_train)
pred_y_test = model.predict(x_test)

r2_train = metrics.r2_score(y_train, pred_y_train)
r2_test = metrics.r2_score(y_test, pred_y_test)

msle_train =metrics.mean_squared_log_error(y_train, pred_y_train)
msle_test =metrics.mean_squared_log_error(y_test, pred_y_test)

print(f"Train r2 = {r2_train:.2f} \nTest r2 = {r2_test:.2f}")
print(f"Train msle = {msle_train:.2f} \nTest msle = {msle_test:.2f}")

print(gs_knn.best_params_)

```

## Deployment

```
On terminal run the following command to start the app on the default port: 5000
flask run

i.e http://127.0.0.1:5000

```
