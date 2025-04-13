from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_file(path):
    return pd.read_csv(path)

def set_col_name(N,data_name):
    if(data_name=="train"):
        new_col_names = ["Case", "Site_of_recording","Population_area", "Sex", "Age", "Head_length", "Skull_weight", "Total_length", "Tail_length", "Foot_length", "Ear_conch_length", "Eye", "Chest", "Belly"]
        N.columns = new_col_names

    if(data_name=="test"):
        new_col_names = ["Case", "Site_of_recording","Population_area", "Sex", "Head_length", "Skull_weight", "Total_length", "Tail_length", "Foot_length", "Ear_conch_length", "Eye", "Chest", "Belly"]
        N.columns = new_col_names

def show_details_col(N,text):
    numeric_cols = ["Case","Head_length", "Skull_weight", "Total_length", "Tail_length", "Foot_length", "Ear_conch_length", "Eye", "Chest", "Belly"]
    # رسم Boxplot برای بررسی مقادیر پرت
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=N[numeric_cols])
    plt.xticks(rotation=45)
    plt.title("Boxplot of Processed Features")
    # نمایش عنوان کلی در بالای پنجره
    plt.suptitle(text, fontsize=14, fontweight='bold')
    plt.show()

def show_feature_importances(model,X_train):
    # دریافت اهمیت ویژگی‌ها
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    # رسم نمودار اهمیت ویژگی‌ها
    feature_importances.sort_values().plot(kind='barh', figsize=(8, 5))
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance in Random Forest Model")
    plt.show()

def scaler(N):
    # لیست ستون‌های عددی
    numeric_cols = ["Case", "Head_length", "Skull_weight", "Total_length", 
                    "Tail_length", "Foot_length", "Ear_conch_length", "Eye", "Chest", "Belly"]

    # استانداردسازی داده‌ها
    scalers = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler()
    }

    # اعمال StandardScaler روی برخی از ویژگی‌ها
    standard_cols = ["Case", "Head_length", "Skull_weight", "Total_length", 
                     "Chest", "Belly", "Tail_length", "Foot_length"]
    N.loc[:, standard_cols] = scalers['Standard'].fit_transform(N[standard_cols])

    # اعمال RobustScaler روی برخی ویژگی‌های دیگر
    robust_cols = ["Ear_conch_length", "Eye"]
    N.loc[:, robust_cols] = scalers['Robust'].fit_transform(N[robust_cols])

    return N

def remove_col(N):
    remove = ["Sex", "Site_of_recording", "Population_area"]
    N = N.drop(columns=remove)
    return N

def remove_outliers_iqr(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data.clip(lower=lower_bound, upper=upper_bound)  # جایگزینی مقادیر پرت

def apply_winsorization(series, limits=(0.05, 0.05)):  
    return pd.Series(winsorize(series, limits=limits), index=series.index)


def normalize(N, text):
    """
    این تابع داده‌ها را تمیزسازی، استانداردسازی و نرمال‌سازی می‌کند.
    
    مراحل:
    1. حذف مقادیر گمشده
    2. استانداردسازی با StandardScaler و RobustScaler
    3. حذف و جایگزینی مقادیر پرت
    4. نمایش جعبه‌ای از داده‌ها برای بررسی تغییرات
    """
    numeric_cols = ["Case", "Head_length", "Skull_weight", "Total_length", 
                    "Tail_length", "Foot_length", "Ear_conch_length", "Eye", "Chest", "Belly"]


    # حذف مقادیر گمشده
    N = N.dropna().copy()  
    N = scaler(N)

    # حذف و جایگزینی مقادیر پرت برای تمامی ستون‌های عددی
    for col in numeric_cols:  
        if col in N.columns:  
            N[col] = remove_outliers_iqr(N[col])  # حذف مقادیر پرت با IQR
            N[col] = apply_winsorization(N[col])  # تنظیم مقادیر پرت باقی‌مانده

    # نمایش اطلاعات پس از نرمال‌سازی
    show_details_col(N, text)

    return N

def Model(X1, X2, y1, y2):
    # ایجاد مدل Random Forest و آموزش آن
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X1, y1)
    y_pred = model.predict(X2)

    # محاسبه معیارهای ارزیابی
    mae = mean_absolute_error(y2, y_pred)
    rmse = np.sqrt(mean_squared_error(y2, y_pred))
    r2 = r2_score(y2, y_pred)

    # ذخیره نتایج در قالب DataFrame
    results_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R² Score"],
        "Value": [mae, rmse, r2]
    })

    # نمایش نتایج
    print(results_df.to_string(index=False))

    return model

def Test(N,model,output_path):
    output = pd.read_csv(output_path)
    X_new = N.copy() 
    output["age"] = model.predict(X_new) 
    output.to_csv("output.csv", index=False)
    print(" پیش‌بینی مقدار age انجام شد و فایل ذخیره گردید.")

train=get_file("train_data.csv")
test=get_file("test_data.csv")
#-------------------------------
set_col_name(train,"train")
set_col_name(test,"test")
#-------------------------------
train=remove_col(train)
test=remove_col(test)
#-------------------------------
train=normalize(train,"train")
test=normalize(test,"test")
#-------------------------------
X = train.drop(columns = ["Age"])
y = train["Age"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0,shuffle=True)
model=Model(X_train, X_test, y_train, y_test)
#-------------------------------
Test(test,model,"test_data.csv")
#-------------------------------
show_feature_importances(model,X_train)