import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
#PART A
##loading the data set
df=pd.read_csv("C:\medical cost personal data.csv") 
#first 5 observations
print(df.head(5))
#dataset information
print(df.info())
#summary statistics
print(df.describe())
print(df.shape)

#3
#the dataset represents medical insurance costs based on demographic and lifestyle factors

#the depedent variable is charges

#the indepedent variables are age,sex,bmi,children,smoker and region


#data cleaning PART B
#checking missing values
print(df.isnull().sum())

#no missing values ,so no handling needed

#checking duplicate rows
print(df.duplicated().sum())
if df.duplicated().sum()>0:
    df=df.drop_duplicates()
    print("duplicates removed.")
else:
    print("no duplicates.")

    #

#outlier analysis
import seaborn as sns
plt.figure()
sns.boxplot(x=df["bmi"])
plt.title("bmi boxplot")
plt.show()
plt.figure()
sns.boxplot(x=df["charges"])
plt.title("boxplot of charges")
plt.show
Q1=df["charges"].quantile(0.25)
Q3=df["charges"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
print("charges IQR Lower bound:;",lower_bound)
print("charges IQR Upper bound:;",upper_bound )

#using the IQR method,no outliers were identified in bmi or charges.even if there were, i would retain them 
#as they may represent valid cases

##encoding and categorical variables
df["sex"]=df["sex"].map({"female":0,"male":1}).astype(float)
df["smoker"]=df["smoker"].map({"no":0,"yes":1}).astype(float)
df=pd.get_dummies(df,columns=["region"],prefix="region")
df[["region_northeast","region_northwest","region_southeast","region_south_west"]]=df[["region_northeast","region_northwest","region_southeast","region_southwest"]].astype(float)
print(df.head())

df_encoded=pd.get_dummies(df,drop_first=True)
print("encoded dataset preview:")
print(df_encoded.head())

#binary mapping for sex(0=female,1=male) and smoker(0=no,1=yes) since they are binary categoies.one _hot encoding(dummy variables)
#for region to create binary columns for each category, avoiding ordinal assumption
##5
#feature scaling is not necessary for linear regression because the algorithm adjusts coefficients to account for different scales.
#scaling is more critical for algorithms like KNN or neural networks, but here it would not change the models perfomance

#exploratory data analysis PART C
#summary statistics

print(df.describe())

#histogram of charges

sns.histplot(df["charges"],kde=True)
plt.title("distribution of medical charges")
plt.show()
#code of skewness
print(df["charges"].skew())

#the distribution of charges is positively skewed ,indicating most individuals have low charges, with a few high_costcases
#3
sns.scatterplot(x=df["age"],y=df["charges"])
plt.title("age vs charges")
plt.show()
plt.figure()
#
sns.scatterplot(x=df["bmi"],y=df["charges"])
plt.title("bmi vs charges")
plt.show()
plt.figure()
#
sns.boxplot(x=df["smoker"],y=df["bmi"])
plt.title("charges by smoking status")
plt.show()
plt.figure(figsize=(10,8))

print(df.corr(numeric_only=True))
print(df["age"].corr(df["charges"]))
print(df["bmi"].corr(df["charges"]))
print(df.groupby("smoker")["charges"].describe())

#smoker appears most strongly correlated to charges

#yes smokers pay more than non smokers since smokers have a higher mean of 32050 and median of 34456

#age and bmi show positive but weak correlations with charges suggesting approximate linearity, but the relationship is stronger for smoker
#the scatter plots would show upward trends for age and bmi with clusters for smokers in higher charges. 
#the boxplot shows smokers with higher and more varied  charges

##simple linear regression PART D
x_simple=df[["age"]]
y=df["charges"]
model_simple=LinearRegression().fit(x_simple,y)
intercept=model_simple.intercept_
slope=model_simple.coef_[0]
r2_simple=model_simple.score(x_simple,y)
print("regression equation:charges={:.2f}+{:.2f}*age".format(intercept,slope))
print("intercept:{:.2f}".format(intercept))
print("slope:{:.2f}".format(slope))
print("r2:{:.4f}".format(r2_simple))

##the intercept 3190.02 is the estimated charge whn age is 0, the slope257.23 means that for each additional year 
#of age, medical charges increase by by 257.23 on average
#r2 of 0.0890 indicates that age explains only 8.90% of the variance in charges, suggesting its not a strong predictor alone



#multiple linear regression
import statsmodels.api as sm
df_encoded=pd.get_dummies(df,drop_first=True)
df_encoded=df_encoded.astype(float)
x=df_encoded.drop("charges",axis=1)
y=df_encoded["charges"]
x=sm.add_constant(x)
model_multiple=sm.OLS(y,x).fit()
print(model_multiple.summary())
#
results_table=pd.DataFrame({
    "variable": model_multiple.params.index,
    "coefficient": model_multiple.bse.values,
    "p_value": model_multiple.pvalues.values
})
results_table=results_table.round(3)

print(results_table)
print("r-squared:", round(model_multiple.rsquared, 3))
print("adjusted r-squared:",round(model_multiple.rsquared_adj, 3))

#variables that are statistically significant
#variables with p<0.05 for example age,bmi,children,smoking staus,region south east,region southwest

#factor with the strongest impact
#smoking status has the strongest impactbecause it has the largest coefficient

#smoking affects medical charges in a way thatholding other variables constant smokers pay more than non smokers
#this indicates that smokers represent significantly higher financial risk


# 9 train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
ir=LinearRegression()
ir.fit(x_train,y_train)
y_pred=ir.predict(x_test)

#10 model valuation
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("model evaluation on test data:")
print("RMSE:",rmse)
print("MAE:",mae)
print("R2:",r2)

#11 regression assumptions
residuals=y_test-y_pred

##
plt.figure()
sns.scatterplot(x=y_pred,y=residuals)
plt.axhline(0,color="red")
plt.title("residual plot")
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.show()
plt.figure()

sns.histplot(residuals,kde=True)
plt.title("residual distribution")
plt.show()
sm.qqplot(residuals,line="45")
plt.title("Q-Q plot")
plt.show()

#

##12 multicollinearity
from statsmodel.stats.outliers_ifluence import variance_inflation_factor
vif_data=pd.DataFrame()
vif_data["Feature"]=x.columns
vif_data["VIF"]=[variance_inflation_factor(x.values,i)
                 for i in range(x.shape[1])]
                 

print(vif_data)


##13interpretation outputs PART F
bmi_coef=model_multiple.params("bmi")
smoker_coef=model_multiple.params("smoker yes")
print(f"if BMI increases by 1 unit,charges increase by approximately{bmi_coef:.2f}.")
print(f"smokers pay approximately{smoker_coef:.2f} more than non smokers on average.")
print("analysis complete")
#RECCOMENDATIONS
#charging higher premiums for smokers to manage financial risk, as smoking increases charges
#offering incetives for mantaining healthy bmi and non smoking as these factors impact charges
#adjust pricing based on age targetting older clients with risk management programmes

##REPORT WRITING






