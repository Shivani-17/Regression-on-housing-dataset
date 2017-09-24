import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import math
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from scipy.stats import skew
from sklearn.preprocessing import StandardScalar

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20


def main():
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')
	plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
	df_train = df_train[df_train['GrLivArea'] < 4000]
	df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
	df_train = missing(df_train)
	df_test = missing(df_test)
	df_train = encode(df_train)
	df_test = encode(df_test)
	df_train = correct_categorical(df_train)	
	df_test = correct_categorical(df_test)
	df = pd.DataFrame(columns  = ['Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 'Electrical_Mix', 'Electrical_No', 'MiscFeature_TenC'])
	#print('train',df_train.shape)
	#print('test',df_test.shape)
	df_test = correct_test(df_test)
	df_test = df_test.join(df)
	df_test = df_test.fillna(0.0)
	#df_train['MasVnrArea'] = df_train['MasVnrArea'].convert_objects(convert_numeric=True)
	#df_test['MasVnrArea'] = df_test['MasVnrArea'].convert_objects(convert_numeric=True)
	#print('train',df_train.shape)
	#print('test',df_test.shape)
	#l = df_test.isnull().sum()
	#print(l[l > 0])
	df_train = create_new_features(df_train)
	df_test = create_new_features(df_test)
	corr = df_train.corr()
	corr.sort_values(['SalePrice'],ascending=False,inplace=True)	
	#print(corr['SalePrice'])
	df_train = add_degree(df_train)
	df_test = add_degree(df_test)
	df_train = correct_skewness(df_train)
	df_test = correct_skewness(df_test)
	model(df_train,df_test)


def correct_skewness(df) :
	num_col = df.select_dtypes(exclude = ['object']).columns
	skewness = df[num_col].apply(lambda x : skew(x))
	skewness = skewness[abs(skewness) > 0.5]
	skewed_features = skewness.index
	df[skewed_features] = np.log1p(df[skewed_features])
	return df

def add_degree(df) :
	df['OverallQual_2'] = df['OverallQual'] ** 2
	df['OverallQual_3'] = df['OverallQual'] **  3
	df['OverallQual_sq'] = np.sqrt(df['OverallQual'])
	df['GrLivArea_2'] = df['GrLivArea'] ** 2
	df['GrLivArea_3']  = df['GrLivArea'] ** 3
	df['GrLivArea_sq'] = np.sqrt(df['GrLivArea'])
	df['GarageCars_2'] = df['GarageCars'] ** 2
	df['GarageCars_3'] = df['GarageCars'] ** 3
	df['GarageCars_sq'] = np.sqrt(df['GarageCars'])
	df['Bath_Tot_2'] = df['Bath_Tot'] ** 2
	df['Bath_Tot_3'] = df['Bath_Tot'] ** 3
	df['Bath_Tot_sq'] = np.sqrt(df['Bath_Tot'])
	df['KitchenQual_2'] = df['KitchenQual'] ** 2
	df['KitchenQual_3'] = df['KitchenQual'] ** 3
	df['KitchenQual_sq'] = np.sqrt(df['KitchenQual'])
	df['GarageArea_2']  = df['GarageArea'] ** 2
	df['BsmtQual_2']  = df['BsmtQual'] ** 2
	df['ExterQual_2']  = df['ExterQual'] ** 2
	df['TotalBsmtSF_2']  = df['TotalBsmtSF'] ** 2
	df['GarageArea_3'] = df['GarageArea'] ** 3
	df['GarageArea_sq'] = np.sqrt(df['GarageArea'])
	df['BsmtQual_3'] = df['BsmtQual'] ** 3
	df['GarageArea_sq'] = np.sqrt(df['GarageArea'])
	df['ExterQual_3'] = df['ExterQual'] ** 3
	df['ExterQual_sq'] = np.sqrt(df['ExterQual'])
	df['TotalBsmtSF_3'] = df['TotalBsmtSF'] ** 3
	df['TotalBsmtSF_sq'] = np.sqrt(df['TotalBsmtSF'])
	return df


def create_new_features(df) :
	df['OverAll_Cond'] = df['OverallQual']	 +  df['OverallCond']
	df['Exter_Cond'] = df['ExterQual'] + df['ExterCond']
	df['Basement_Ar'] = df['BsmtQual'] + df['BsmtFinType1'] + df['BsmtExposure'] + df['BsmtFinType2'] + df['BsmtCond']
	df['Bath_Tot'] = df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'] + df['FullBath'] + 0.5*df['HalfBath']
	df['Garage_Tot']  = df['GarageFinish'] + df['GarageQual'] + df['GarageCond']
	df["PoolScore"] = df["PoolArea"] + df["PoolQC"]	
	return df


def correct_test(df) :
	items = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','GarageCars'
,'GarageArea']
	for item in items :
		median = df[item].median()
		df[item] = df[item].fillna(median)
	return df
	
def correct_categorical(df) :
	qualitative = [col for col in df.columns if df.dtypes[col] == 'object']
	dummies = pd.get_dummies(df[[col for col in qualitative]])
	#print('BEFORE ',df.shape)
	df = df.drop([col for col in qualitative], axis = 1)
	df = df.join(dummies)
	#print('CHECK',df.shape)
	return df


def missing(df) :
	qualitative = [col for col in df.columns if df.dtypes[col] == 'object']
	quantitative = [col for col in df.columns if df.dtypes[col] != 'object']
	missing_qualitative = df[[col for col in qualitative]].isnull().sum()
	missing_qualitative = missing_qualitative[ missing_qualitative > 0]	
	missing_quantitative = df[[col for col in quantitative]].isnull().sum()
	missing_quantitative = missing_quantitative[ missing_quantitative > 0]
	df = correct_missing(missing_quantitative,df)
	df = correct_missing_qual(missing_qualitative,df)
	return df

def linear_regression(y_train,x_train,df_test) :
	clf = LinearRegression()
	clf.fit(x_train,y_train)
	pred = clf.predict(df_test)
	acc_log = round(clf.score(x_train, y_train) * 100, 2)
	print(acc_log)
	return pred


def ridge_regression(y_train,x_train,df_test) :
	ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
	ridge.fit(x_train, y_train)	
	alpha = ridge.alpha_
	ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
	ridge.fit(x_train, y_train)
	alpha = ridge.alpha_
	print('ALPHA', alpha)
	acc_log = round(ridge.score(x_train, y_train) * 100, 2)
	print('SCORE' , acc_log)
	pred = ridge.predict(df_test)
	return pred

def lasso_regression(y_train,x_train,df_test) :
	lasso = LassoCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
	lasso.fit(x_train, y_train)	
	alpha = lasso.alpha_
	ridge = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
	lasso.fit(x_train, y_train)
	alpha = lasso.alpha_
	print('ALPHA', alpha)
	#acc_log = round(ridge.score(x_train, y_train) * 100, 2)
	#print('SCORE' , acc_log)
	pred = lasso.predict(df_test)
	return pred

def model(df_train,df_test) :
	df_train['SalePrice'] = df_train['SalePrice'].astype(float)
	y_train = df_train.SalePrice 
	x_train = df_train.drop('SalePrice',axis = 1)
	#pred = linear_regression(y_train,x_train,df_test)
	pred = ridge_regression(y_train,x_train,df_test)
	#pred = lasso_regression(y_train,x_train,df_test)
	for item in pred :
		print(item)
	for i in range(0,len(pred)) :
		pred[i] = math.exp(pred[i]) -1
	submission = pd.DataFrame({
	"Id": df_test["Id"],
	"SalePrice": pred
	})
	submission.to_csv('out.csv', index=False)

def encode(df_train) :
	df_train = df_train.replace({'Alley' : {'Pave' : 0,'Gravel' : 1},
					'BsmtCond' : {'TA' : 3, 'Gd' : 4, 'No' : 0, 'Fa' : 2, 'Po' : 1},
					'BsmtExposure' : {'No' : 0, 'Gd' : 3, 'Mn':1, 'Av':2},
				'BsmtFinType1' : {'GLQ' : 6, 'ALQ' : 5, 'Unf' : 1, 'Rec' : 3, 'BLQ' : 4, 'No' : 0, 'LwQ' :2 },
				'BsmtFinType2' : {'GLQ' : 6, 'ALQ' : 5, 'Unf' : 1, 'Rec' : 3, 'BLQ' : 4, 'No' : 0, 'LwQ' :2},
				'BsmtQual' : {'Gd' : 3, 'TA' : 2, 'Ex' : 4, 'No' : 0, 'Fa' : 1},
				'CentralAir' : {'Y' : 1,'N' : 0},
				'ExterCond' : {'TA' : 1, 'Gd' : 3, 'Fa' : 2, 'Po' : 0, 'Ex' : 4},
				'ExterQual' : {'Gd': 2, 'TA' : 0, 'Ex' : 3, 'Fa' : 1},	
				'Fence' : {'No' : 0, 'MnPrv' : 3, 'GdWo' : 2, 'GdPrv' : 4, 'MnWw' : 1},
				 'FireplaceQu' : {'No' : 0, 'TA' : 3, 'Gd' : 4, 'Fa' :2, 'Ex' : 5, 'Po' : 1},			
				'Functional' : {'Typ' : 6, 'Min1' : 5, 'Maj1' : 2, 'Min2' : 4, 'Mod' : 3, 'Maj2' : 1, 'Sev' : 0},
				'GarageCond' :  {'TA' : 2, 'Fa' : 3, 'No' : 0, 'Gd' : 4, 'Po' : 1, 'Ex' : 5},
				'GarageFinish': {'RFn':2, 'Unf':1, 'Fin' : 3, 'No' : 0},
				'GarageQual' : {'TA' : 2, 'Fa' : 3, 'Gd' : 4, 'No' : 0, 'Ex' : 5, 'Po' : 1},	
				'HeatingQC' : {'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0},
			'KitchenQual' : {'Gd' : 2,'TA' : 1, 'Ex' : 3, 'Fa' :0 },
			'LotShape' : 	{'Reg' : 3, 'IR1' : 2, 'IR2' :1 , 'IR3' : 0},
			 'PavedDrive' : {'Y' : 2, 'N' : 0, 'P' : 1},
				'PoolQC' : {'None' : 0, 'Ex' : 4, 'Fa' : 1,  'Gd' : 3}
					})
		
	return df_train



def visualize(df_train) :
	sns.boxplot(df_train['Street'],df_train['SalePrice'])
	plt.ylim(0,50)
	plt.show()
	#print(df_train['Street'])

def check_missing(df_train) :
	missing = df_train.isnull().sum()
	missing = missing[missing > 0]
	print(missing)

def correct_missing(missing_columns,df):
	for item in missing_columns.index :
		median = df[item].median()
		df[item] = df[item].fillna(median)	
		return df

def correct_missing_qual(missing_qualitative,df_train) :
	df_train.loc[:,'Alley'] = df_train.loc[:,'Alley'].fillna("None")
	df_train.loc[:,'GarageYrBlt'] = df_train.loc[:,'GarageYrBlt'].fillna(0)
	df_train.loc[:,'MasVnrType']  =  df_train.loc[:,'MasVnrType'].fillna("None")
	df_train.loc[:,'MasVnrArea']  =  df_train.loc[:,'MasVnrArea'].fillna(0.0)
	df_train.loc[:,'BsmtQual']= df_train.loc[:,'BsmtQual'].fillna("No")
	df_train.loc[:,'BsmtCond']= df_train.loc[:,'BsmtCond'].fillna("No")
	df_train.loc[:,'BsmtExposure']= df_train.loc[:,'BsmtExposure'].fillna("No")
	df_train.loc[:,'BsmtFinType1']= df_train.loc[:,'BsmtFinType1'].fillna("No")
	df_train.loc[:,'BsmtFinType2']= df_train.loc[:,'BsmtFinType2'].fillna("No")
	df_train.loc[:,'PoolQC']= df_train.loc[:,'PoolQC'].fillna("None")
	df_train.loc[:,'FireplaceQu']= df_train.loc[:,'FireplaceQu'].fillna("No")
	df_train.loc[:,'GarageType']= df_train.loc[:,'GarageType'].fillna("No")
	df_train.loc[:,'GarageFinish']= df_train.loc[:,'GarageFinish'].fillna("No")
	df_train.loc[:,'PoolQC']= df_train.loc[:,'PoolQC'].fillna("No")
	df_train.loc[:,'GarageQual']= df_train.loc[:,'GarageQual'].fillna("No")
	df_train.loc[:,'PoolQC']= df_train.loc[:,'PoolQC'].fillna("No")
	df_train.loc[:,'GarageCond']= df_train.loc[:,'GarageCond'].fillna("No")
	df_train.loc[:,'Fence']= df_train.loc[:,'Fence'].fillna("No")
	df_train.loc[:,'MiscFeature']= df_train.loc[:,'MiscFeature'].fillna("No")
	df_train.loc[:,'Electrical']= df_train.loc[:,'Electrical'].fillna("No")
	return df_train			  



main()
