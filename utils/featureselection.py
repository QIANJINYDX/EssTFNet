from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

dataset="../features.csv"
D = pd.read_csv(dataset)  # Using R

# print('Before drop duplicates: {}'.format(D.shape))
# D = D.drop_duplicates()  # Return : each row are unique value
# print('After drop duplicates: {}\n'.format(D.shape))

X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X,y=x_test,y_test

# use linear regression as the model
lr = LinearRegression()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X, y)
with open('../data/featuresselection.pkl', 'wb') as f:
    pickle.dump(list(rfe.ranking_), f)
print("Features sorted by their rank:")
print(rfe.ranking_)
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))