from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from math import ceil, floor

# load house data
df = pd.read_csv(r"./data/house_price.csv")

# Drop the following columns
dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1",
                   "Condition2", "BldgType", "OverallCond", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                   "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath",
                   "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal",
                   "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch",
                   "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType",
                   "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)

droppedDf.isnull().sum().sort_values(ascending=False)
droppedDf["Alley"].fillna("NO", inplace=True)
droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
droppedDf["GarageFinish"].fillna("NO", inplace=True)
droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
droppedDf["BsmtQual"].fillna("NO", inplace=True)
droppedDf["MasVnrArea"].fillna(0, inplace=True)
droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea > 1000, 'BIG',
                                    np.where(droppedDf.MasVnrArea > 500, 'MEDIUM',
                                    np.where(droppedDf.MasVnrArea > 0, 'SMALL', 'NO')))

droppedDf = droppedDf.drop(['SalePrice'], axis=1)

# load the model weights and predict the target
modelName = r"./model/trained_model.model"
loaded_model = pickle.load(open(modelName, 'rb'))


# create flask app instance
app = Flask(__name__)


@app.route("/api", methods=['GET', 'POST'])
def predict():
    # get data from request
    print('Got HERE!!! @ /api')
    data = request.get_json(force=True)
   
    inputDf = droppedDf.iloc[[0]].copy()

    for i in inputDf:
        if inputDf[i].dtype == "object":
            inputDf[i] = droppedDf[i].mode()[0]
        elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
            inputDf[i] = droppedDf[i].mean()

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    floatobj_feat = list(inputDf.loc[:, inputDf.dtypes == 'float64'].columns.values)

    requestDataDict = data
    print('RequestData Dict: {}'.format(requestDataDict))

    droppedList = ["LotShape", "LotConfig", "BsmtQual", "HeatingQC", "CentralAir", "KitchenQual", "GarageFinish", "FullBath", "Alley"]

    inputDfTest = inputDf

    for key in droppedList:
        if key not in requestDataDict.keys():
            requestDataDict[key] = inputDfTest[key]

    inputDf = pd.DataFrame(requestDataDict, index=[0])
    # print('inputDf2: {}'.format(inputDf))


    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    for feature in floatobj_feat:
        inputDf[feature] = inputDf[feature].astype('int')
    
    #Make prediction using model
    prediction = loaded_model.predict(inputDf)

    return Response(json.dumps(prediction[0]))