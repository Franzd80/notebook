from fastapi import FastAPI, File, UploadFile
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler  

# Crea un'app FastAPI
app = FastAPI()

# Carica il modello e lo scaler salvati
model = joblib.load('house_price.pkl')
scaler = joblib.load('scaler.pkl')  # Assicurati di avere il file scaler.pkl

# Definisci il formato dei dati in ingresso usando Pydantic
class HousePricingInput(BaseModel):
    Id: int
    MSSubClass: int
    MSZoning: int
    LotFrontage: float = None  # Gestisci i valori nulli
    LotArea: int
    Alley: int
    LotShape: int
    LandContour: int
    LotConfig: int
    LandSlope: int
    Neighborhood: int
    Condition1: int
    Condition2: int
    BldgType: int
    HouseStyle: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: int
    RoofMatl: int
    Exterior1st: int
    Exterior2nd: int
    MasVnrType: int
    MasVnrArea: float = None
    ExterQual: int
    ExterCond: int
    Foundation: int
    BsmtQual: int
    BsmtCond: int
    BsmtExposure: int
    BsmtFinType1: int
    BsmtFinSF1: float = None
    BsmtFinType2: int
    BsmtFinSF2: float = None
    BsmtUnfSF: float = None
    TotalBsmtSF: float = None
    Heating: int
    HeatingQC: int
    CentralAir: int
    Electrical: int
    FirstFlrSF: int
    SecondFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: float = None
    BsmtHalfBath: float = None
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: int
    TotRmsAbvGrd: int
    Functional: int
    Fireplaces: int
    FireplaceQu: int
    GarageType: int
    GarageYrBlt: float = None
    GarageFinish: int
    GarageCars: float = None
    GarageArea: float = None
    GarageQual: int
    GarageCond: int
    PavedDrive: int
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: int
    Fence: int
    MiscFeature: int
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: int
    SaleCondition: int

# Definisci il punto di accesso per le predizioni usando il JSON
@app.post("/predict")
def predict(input_data: HousePricingInput):
    # Converte i dati in array numpy usando un elenco di tutte le feature
    data = np.array([[
        input_data.Id, input_data.MSSubClass, input_data.MSZoning, input_data.LotFrontage, input_data.LotArea, input_data.Alley,
        input_data.LotShape, input_data.LandContour, input_data.LotConfig, input_data.LandSlope, input_data.Neighborhood,
        input_data.Condition1, input_data.Condition2, input_data.BldgType, input_data.HouseStyle, input_data.OverallQual,
        input_data.OverallCond, input_data.YearBuilt, input_data.YearRemodAdd, input_data.RoofStyle, input_data.RoofMatl,
        input_data.Exterior1st, input_data.Exterior2nd, input_data.MasVnrType, input_data.MasVnrArea, input_data.ExterQual,
        input_data.ExterCond, input_data.Foundation, input_data.BsmtQual, input_data.BsmtCond, input_data.BsmtExposure,
        input_data.BsmtFinType1, input_data.BsmtFinSF1, input_data.BsmtFinType2, input_data.BsmtFinSF2, input_data.BsmtUnfSF,
        input_data.TotalBsmtSF, input_data.Heating, input_data.HeatingQC, input_data.CentralAir, input_data.Electrical,
        input_data.FirstFlrSF, input_data.SecondFlrSF, input_data.LowQualFinSF, input_data.GrLivArea, input_data.BsmtFullBath,
        input_data.BsmtHalfBath, input_data.FullBath, input_data.HalfBath, input_data.BedroomAbvGr, input_data.KitchenAbvGr,
        input_data.KitchenQual, input_data.TotRmsAbvGrd, input_data.Functional, input_data.Fireplaces, input_data.FireplaceQu,
        input_data.GarageType, input_data.GarageYrBlt, input_data.GarageFinish, input_data.GarageCars, input_data.GarageArea,
        input_data.GarageQual, input_data.GarageCond, input_data.PavedDrive, input_data.WoodDeckSF, input_data.OpenPorchSF,
        input_data.EnclosedPorch, input_data.ThreeSsnPorch, input_data.ScreenPorch, input_data.PoolArea, input_data.PoolQC,
        input_data.Fence, input_data.MiscFeature, input_data.MiscVal, input_data.MoSold, input_data.YrSold, input_data.SaleType,
        input_data.SaleCondition
    ]])

    # Applica lo scaler ai dati
    scaled_data = scaler.transform(data)

    # Effettua la predizione
    prediction = model.predict(scaled_data)

    # Restituisci il risultato come JSON
    return {"prediction": int(prediction[0])}

# Definisci il punto di accesso per il caricamento del CSV
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # Leggi il file CSV in un dataframe pandas
    df = pd.read_csv(file.file)

    # Verifica che il file contenga tutte le colonne richieste
    expected_columns = [
        'Id','MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 
        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 
        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 
        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 
        'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 
        'YrSold', 'SaleType', 'SaleCondition'
    ]
    
    # Verifica se il CSV contiene tutte le colonne necessarie
    if not all(column in df.columns for column in expected_columns):
        return {"error": "Il file CSV non contiene tutte le feature necessarie."}

    # Estrai le feature richieste dal dataframe
    data = df[expected_columns].values

    # Applica lo scaler ai dati
    x_test_new = scaler.transform(data)

    # Fai le predizioni per tutte le righe nel CSV
    predictions = model.predict(x_test_new)

    # Restituisci le predizioni come JSON
    return {"predictions": predictions.tolist()}
