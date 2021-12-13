from numpy import NaN
import pandas as pd
from pandas._libs.missing import NA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import parser

df_fifa = pd.read_csv("./data/fifa.csv")
df_fifa = (df_fifa
    .drop("ID", axis="columns")
    .drop("Name", axis="columns")
    .drop("Photo", axis="columns")
    .drop("Nationality", axis="columns")
    .drop("Flag", axis="columns")
    .drop("Club", axis="columns")
    .drop("Club Logo", axis="columns")
    )

def transform_currency_to_absolute(valueString):
    range = valueString[-1]
    # print("Range: ", range)
    if(range == "M"):
        val = float(valueString[1:-1])
        # print("Value: ", val)
        return val * 1_000_000
    elif(range == "K"):
        val = float(valueString[1:-1])
        # print("Value: ", val)
        return val * 1_000
    elif(range == "0"):
        return float(0)
    return NaN

df_fifa["Value"] = df_fifa["Value"].apply(lambda valueString: transform_currency_to_absolute(valueString))
df_fifa["Wage"] = df_fifa["Wage"].apply(lambda valueString: transform_currency_to_absolute(valueString))

positions_set = set()
for positions_array in df_fifa["Preferred Positions"].apply(lambda positions: positions.split(" ")):
    for position in positions_array:
        if(position):
            positions_set.add(position)

for position in positions_set:
    df_fifa["Preferred Position " + position] = df_fifa["Preferred Positions"].apply(lambda pos: pos.split(" ")).apply(lambda pos_arr: position in pos_arr)
    df_fifa["Preferred Position " + position] = df_fifa["Preferred Position " + position].astype(int)

df_fifa = df_fifa.drop("Preferred Positions", axis="columns")

def clean_object_values_to_string(value):
    if(value is float):
        return value
    elif(value is int):
        return float(value)
    else:
        try:
            code = parser.expr(str(value)).compile()
            return float(eval(code))
        except:
            print("Error occured when parsing and cleaning ", value, ". Replacing val with NaN.")
            return NA


for key, value in df_fifa.dtypes.items():
    if(value == object):
        df_fifa[key] = df_fifa[key].apply(lambda val: clean_object_values_to_string(val))

df_fifa = df_fifa.dropna()

y = df_fifa.copy(deep=True)["Value"]
X = df_fifa.copy(deep=True).drop("Value", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

reg = LassoCV(cv=5, random_state=0).fit(X, y)
y_pred = reg.predict(X_test)
print("Test set performance: ", mse(y_true=y_test, y_pred=y_pred))

df_fifa.to_csv("./data/fifa_preprocessed.csv")