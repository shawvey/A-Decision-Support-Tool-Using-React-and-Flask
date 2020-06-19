from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import pandas as pd

dataset = pd.read_csv('data.csv')
fill_knn = KNN(k=5).fit_transform(dataset)
data = pd.DataFrame(fill_knn)
data.to_csv('imputated_data.csv',index=False,header=True)
