from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from math import ceil, floor

# load house data
df = pd.read_csv(r"./data/house_price.csv")

# Drop the following columns
dropColumns = ["Id", "Alley", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd","MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1", "LotShape", "LotConfig", "BsmtQual", "HeatingQC", "CentralAir", "KitchenQual", "GarageFinish", "FullBath",
              "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)

droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
# droppedDf["GarageFinish"].fillna("NO", inplace=True)
droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
# droppedDf["BsmtQual"].fillna("NO", inplace=True)
droppedDf["MasVnrArea"].fillna(0, inplace=True)
droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea > 1000, 'BIG',
                                    np.where(droppedDf.MasVnrArea > 500, 'MEDIUM',
                                    np.where(droppedDf.MasVnrArea > 0, 'SMALL', 'NO')))

droppedDf = droppedDf.drop(['SalePrice'], axis=1)
inputDf = droppedDf.iloc[[0]].copy()

for i in inputDf:
    if inputDf[i].dtype == "object":
        inputDf[i] = droppedDf[i].mode()[0]
    elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
        inputDf[i] = droppedDf[i].mean()

obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
floatobj_feat = list(inputDf.loc[:, inputDf.dtypes == 'float64'].columns.values)
intobj_feat = list(inputDf.loc[:, inputDf.dtypes == 'int64'].columns.values)

for feature in obj_feat:
    inputDf[feature] = inputDf[feature].astype('category')

# load the model weights and predict the target
modelName = r"./model/final_model1.model"
loaded_model = pickle.load(open(modelName, 'rb'))


# create flask app instance
app = Flask(__name__)


@app.route("/api", methods=['GET', 'POST'])
def predict():
    # get data from request
    print('Got HERE!!! @ /api')
    data = request.get_json(force=True)
    # requestData = np.array([data["LotShape"], data["LotConfig"], data["Neighborhood"], data["HouseStyle"], data["ExterQual"], data["BsmtQual"],
    #                         data["HeatingQC"], data["CentralAir"], data["KitchenQual"], data["GarageFinish"], data["LotFrontage"], data["LotArea"],
    #                         data["OverallQual"], data["YearBuilt"], data["YearRemodAdd"], data["MasVnrArea"], data["BsmtFinSF1"], data["TotalBsmtSF"],
    #                         data["1stFlrSF"], data["2ndFlrSF"], data["GrLivArea"], data["FullBath"], data["TotRmsAbvGrd"], data["Fireplaces"],
    #                         data["GarageYrBlt"], data["GarageCars"], data["GarageArea"], data["MasVnrAreaCatg"]])

    requestDataDict = data
    print('RequestData Dict: {}'.format(requestDataDict))

    inputDf = pd.DataFrame(requestDataDict, index=[0])
    print('inputDf2: {}'.format(inputDf))


    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    for feature in floatobj_feat:
        inputDf[feature] = inputDf[feature].astype('float')
    
    #Make prediction using model
    prediction = loaded_model.predict(inputDf)

    return Response(json.dumps(prediction[0]))