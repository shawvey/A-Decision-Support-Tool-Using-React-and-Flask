from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import pandas as pd

dataset = pd.read_csv('newdata.csv')
fill_knn = KNN(k=5).fit_transform(dataset)
data = pd.DataFrame(fill_knn)
data.to_csv('imputated_newdata.csv',index=False,header=True)
