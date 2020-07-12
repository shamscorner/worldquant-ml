%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 03:45:19
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 03:45:19
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# Sun, 12 Jul 2020 03:45:23
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])# Sun, 12 Jul 2020 04:22:38
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper# Sun, 12 Jul 2020 04:22:39
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self# Sun, 12 Jul 2020 04:30:49
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))# Sun, 12 Jul 2020 04:35:36
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 04:35:46
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))# Sun, 12 Jul 2020 04:36:31
replacer = OutlierReplacer(5, 95)
replacer.fit(X) # Sun, 12 Jul 2020 04:44:12
X.shape[-1]# Sun, 12 Jul 2020 04:44:20
X.shape# Sun, 12 Jul 2020 04:44:29
X.shape[-1]# Sun, 12 Jul 2020 04:46:07
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt# Sun, 12 Jul 2020 04:48:20
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);# Sun, 12 Jul 2020 05:02:23
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 05:10:01
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass# Sun, 12 Jul 2020 05:10:03
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self# Sun, 12 Jul 2020 05:10:05
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])# Sun, 12 Jul 2020 05:10:07
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))# Sun, 12 Jul 2020 05:10:17
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])# Sun, 12 Jul 2020 05:10:19
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))# Sun, 12 Jul 2020 05:10:50
X.shape# Sun, 12 Jul 2020 05:11:01
X.shape[0]# Sun, 12 Jul 2020 05:11:11
np.ones(X.shape[0])# Sun, 12 Jul 2020 05:11:25
2.068558169089147 * np.ones(X.shape[0])# Sun, 12 Jul 2020 05:13:40
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])# Sun, 12 Jul 2020 05:13:42
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))# Sun, 12 Jul 2020 05:44:39
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 05:46:53
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)# Sun, 12 Jul 2020 05:50:28
X# Sun, 12 Jul 2020 05:50:37
X.shape# Sun, 12 Jul 2020 05:50:56
X[:, -2:]# Sun, 12 Jul 2020 05:51:08
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])# Sun, 12 Jul 2020 05:52:15
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])# Sun, 12 Jul 2020 05:52:32
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])# Sun, 12 Jul 2020 05:56:01
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 05:59:22
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
X.shape
X.shape[0]
np.ones(X.shape[0])
2.068558169089147 * np.ones(X.shape[0])
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)
X
X.shape
X[:, -2:]
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 06:01:05
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))# Sun, 12 Jul 2020 06:01:23
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df# Sun, 12 Jul 2020 06:01:29
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))# Sun, 12 Jul 2020 06:01:36
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])# Sun, 12 Jul 2020 06:01:50
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))# Sun, 12 Jul 2020 06:01:57
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])# Sun, 12 Jul 2020 06:02:01
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))# Sun, 12 Jul 2020 06:03:27
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 06:03:28
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:04:07
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:04:32
pipe.fit(X, y)# Sun, 12 Jul 2020 06:06:28
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:06:31
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 06:06:34
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:06:34
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:08:19
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
X.shape
X.shape[0]
np.ones(X.shape[0])
2.068558169089147 * np.ones(X.shape[0])
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)
X
X.shape
X[:, -2:]
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 06:08:33
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 06:08:35
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:08:35
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:08:37
pipe.fit(X, y)# Sun, 12 Jul 2020 06:09:03
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
X.shape
X.shape[0]
np.ones(X.shape[0])
2.068558169089147 * np.ones(X.shape[0])
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)
X
X.shape
X[:, -2:]
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 06:09:03
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# Sun, 12 Jul 2020 06:09:03
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])# Sun, 12 Jul 2020 06:09:03
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)# Sun, 12 Jul 2020 06:09:03
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))# Sun, 12 Jul 2020 06:09:03
print("β_0: {}".format(model.intercept_))

for i in range(8):
    print("β_{}: {}".format(i+1, model.coef_[i]))# Sun, 12 Jul 2020 06:09:03
print("R^2: {:g}".format(model.score(X, y)))# Sun, 12 Jul 2020 06:09:03
from sklearn.ensemble import GradientBoostingRegressor

# create model and train/fit
model = GradientBoostingRegressor()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("R^2: {:g}".format(model.score(X, y)))# Sun, 12 Jul 2020 06:09:05
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df# Sun, 12 Jul 2020 06:09:05
from sklearn.compose import ColumnTransformer

col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('scaler', StandardScaler(), slice(0,6)) # first 6 columns
    ]
)

col_transformer.fit(X)
Xt = col_transformer.transform(X)

print('MedInc mean before transformation?', X.mean(axis=0)[0])
print('MedInc mean after transformation?', Xt.mean(axis=0)[0], '\n')

print('Longitude mean before transformation?', X.mean(axis=0)[-1])
print('Longitude mean after transformation?', Xt.mean(axis=0)[-1])# Sun, 12 Jul 2020 06:09:05
col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('remove', 'drop', 0),
        ('scaler', StandardScaler(), slice(1,6))
    ]
)

Xt = col_transformer.fit_transform(X)

print('Number of features in X:', X.shape[1])
print('Number of features Xt:', Xt.shape[1])# Sun, 12 Jul 2020 06:09:05
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])# Sun, 12 Jul 2020 06:09:05
pipe.named_steps# Sun, 12 Jul 2020 06:09:05
# fit/train model and predict labels
pipe.fit(X, y)
y_pred = pipe.predict(X)

print(y_pred)
print("R^2: {}".format(pipe.score(X, y)))# Sun, 12 Jul 2020 06:09:05
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))# Sun, 12 Jul 2020 06:09:05
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper# Sun, 12 Jul 2020 06:09:05
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self# Sun, 12 Jul 2020 06:09:05
replacer = OutlierReplacer(5, 95)
replacer.fit(X) # Sun, 12 Jul 2020 06:09:05
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))# Sun, 12 Jul 2020 06:09:05
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt# Sun, 12 Jul 2020 06:09:05
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);# Sun, 12 Jul 2020 06:09:05
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass# Sun, 12 Jul 2020 06:09:05
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self# Sun, 12 Jul 2020 06:09:05
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])# Sun, 12 Jul 2020 06:09:05
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))# Sun, 12 Jul 2020 06:09:05
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 06:09:05
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:09:05
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:09:05
pipe.fit(X, y)# Sun, 12 Jul 2020 06:09:25
X.shape# Sun, 12 Jul 2020 06:09:38
DropColumns([0, 1, 2, 3, 4, 5])# Sun, 12 Jul 2020 06:10:01
DropColumns([6, 7])# Sun, 12 Jul 2020 06:10:08
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([6, 7])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:10:16
pipe.fit(X, y)# Sun, 12 Jul 2020 06:10:31
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:10:32
DropColumns([6, 7])# Sun, 12 Jul 2020 06:11:07
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
print(dist_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:11:32
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
print(dist_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:11:46
#pipe.fit(X, y)# Sun, 12 Jul 2020 06:12:04
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])
print(drop)
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:14:31
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])

union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:14:44
pipe.fit(X, y)# Sun, 12 Jul 2020 06:15:13
# pipe.fit(X, y)# Sun, 12 Jul 2020 06:17:23
from sklearn.datasets import load_wine

data = load_wine()
data# Sun, 12 Jul 2020 06:30:00
from sklearn.datasets import load_wine

data = load_wine()
X = data['data']
y = data['target']# Sun, 12 Jul 2020 06:30:16
X.shape# Sun, 12 Jul 2020 06:30:25
y# Sun, 12 Jul 2020 06:33:50
from Collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)# Sun, 12 Jul 2020 06:34:14
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self# Sun, 12 Jul 2020 06:34:40
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self# Sun, 12 Jul 2020 06:35:14
cls = MajorityClassifier()
cls.fit(X, y)# Sun, 12 Jul 2020 06:37:15
c# Sun, 12 Jul 2020 06:37:30
c = Counter(y)
c# Sun, 12 Jul 2020 06:37:46
c = Counter(y)
c.most_common(1)# Sun, 12 Jul 2020 06:37:59
c = Counter(y)
c.most_common(1)[0]# Sun, 12 Jul 2020 06:38:05
c = Counter(y)
c.most_common(1)[0][0]# Sun, 12 Jul 2020 06:40:31
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])# Sun, 12 Jul 2020 06:40:33
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)# Sun, 12 Jul 2020 06:41:21
mc.mode_# Sun, 12 Jul 2020 06:41:32
cls.mode_# Sun, 12 Jul 2020 06:45:24
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
X.shape
X.shape[0]
np.ones(X.shape[0])
2.068558169089147 * np.ones(X.shape[0])
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)
X
X.shape
X[:, -2:]
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
print("β_0: {}".format(model.intercept_))

for i in range(8):
    print("β_{}: {}".format(i+1, model.coef_[i]))
print("R^2: {:g}".format(model.score(X, y)))
from sklearn.ensemble import GradientBoostingRegressor

# create model and train/fit
model = GradientBoostingRegressor()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("R^2: {:g}".format(model.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.compose import ColumnTransformer

col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('scaler', StandardScaler(), slice(0,6)) # first 6 columns
    ]
)

col_transformer.fit(X)
Xt = col_transformer.transform(X)

print('MedInc mean before transformation?', X.mean(axis=0)[0])
print('MedInc mean after transformation?', Xt.mean(axis=0)[0], '\n')

print('Longitude mean before transformation?', X.mean(axis=0)[-1])
print('Longitude mean after transformation?', Xt.mean(axis=0)[-1])
col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('remove', 'drop', 0),
        ('scaler', StandardScaler(), slice(1,6))
    ]
)

Xt = col_transformer.fit_transform(X)

print('Number of features in X:', X.shape[1])
print('Number of features Xt:', Xt.shape[1])
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
pipe.named_steps
# fit/train model and predict labels
pipe.fit(X, y)
y_pred = pipe.predict(X)

print(y_pred)
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
X.shape
DropColumns([0, 1, 2, 3, 4, 5])
DropColumns([6, 7])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([6, 7])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
DropColumns([6, 7])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
print(dist_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
print(dist_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
#pipe.fit(X, y)
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])
print(drop)
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])

union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
# pipe.fit(X, y)
from sklearn.datasets import load_wine

data = load_wine()
data
from sklearn.datasets import load_wine

data = load_wine()
X = data['data']
y = data['target']
X.shape
y
from Collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self
cls = MajorityClassifier()
cls.fit(X, y)
c
c = Counter(y)
c
c = Counter(y)
c.most_common(1)
c = Counter(y)
c.most_common(1)[0]
c = Counter(y)
c.most_common(1)[0][0]
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)
mc.mode_
cls.mode_
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sun, 12 Jul 2020 06:52:55
X.shape# Sun, 12 Jul 2020 06:52:59
X.shape[0]# Sun, 12 Jul 2020 06:53:21
np.ones(X.shape[0], 1)# Sun, 12 Jul 2020 06:53:44
np.ones((X.shape[0]), 1)# Sun, 12 Jul 2020 06:54:18
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# Sun, 12 Jul 2020 06:54:26
np.ones((X.shape[0]), 1)# Sun, 12 Jul 2020 06:54:44
np.ones(3, 1)# Sun, 12 Jul 2020 06:55:01
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0]), 1), self.fraction.reshape(1, 3))# Sun, 12 Jul 2020 06:55:03
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)# Sun, 12 Jul 2020 06:55:24
cls.predict_proba(X)# Sun, 12 Jul 2020 06:56:17
np.ones((X.shape[0], 1))# Sun, 12 Jul 2020 06:56:46
cls.predict_proba(X)# Sun, 12 Jul 2020 06:56:50
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0], 1)), self.fraction.reshape(1, 3))# Sun, 12 Jul 2020 06:56:53
cls.predict_proba(X)# Sun, 12 Jul 2020 06:57:08
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0], 1)), self.fraction.reshape(1, 3))# Sun, 12 Jul 2020 06:57:09
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)# Sun, 12 Jul 2020 06:57:10
cls.mode_# Sun, 12 Jul 2020 06:57:12
cls.predict_proba(X)# Sun, 12 Jul 2020 06:58:07
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist# Sun, 12 Jul 2020 06:58:07
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]# Sun, 12 Jul 2020 06:58:08
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])

union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])# Sun, 12 Jul 2020 06:58:12
pipe.fit(X, y)# Sun, 12 Jul 2020 06:58:32
# pipe.fit(X, y)# Sun, 12 Jul 2020 09:35:10
cls.predict_proba(X)[0]# Sun, 12 Jul 2020 09:47:55
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC# Sun, 12 Jul 2020 09:48:27
data = fetch_california_housing()
X = data['data']
y = data['target']# Sun, 12 Jul 2020 09:49:50
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR# Sun, 12 Jul 2020 09:50:19
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('rgr', LinearSVR())
])
pipe.fit(X, y)
print(pipe.score(X, y))# Sun, 12 Jul 2020 09:50:34
data = fetch_california_housing()
X = data['data']
y = data['target']# Sun, 12 Jul 2020 09:50:35
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('rgr', LinearSVR())
])
pipe.fit(X, y)
print(pipe.score(X, y))# Sun, 12 Jul 2020 09:51:38
import dill# Sun, 12 Jul 2020 09:52:05
with open('my_model.dill', 'wb') as f:
    dill.dump(pipe, f)# Sun, 12 Jul 2020 09:52:23
!ls -alh my_model.dill# Sun, 12 Jul 2020 09:56:06
# restore from external dill file
with open('my_model.dill', 'rb') as f:
    pipe = dill.load(f)# Sun, 12 Jul 2020 09:56:14
pipe.score(X, y)# Sun, 12 Jul 2020 09:58:02
# if we need to compress the data
import gzip 
with gzip.open('my_model.dill.gz', 'wb') as f:
    dill.dump(pipe, f)# Sun, 12 Jul 2020 09:58:16
!ls -alh my_model.dill.gz# Sun, 12 Jul 2020 10:11:45
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
X.shape[-1]
X.shape
X.shape[-1]
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        print(self.y_mean)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
X.shape
X.shape[0]
np.ones(X.shape[0])
2.068558169089147 * np.ones(X.shape[0])
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X)
X
X.shape
X[:, -2:]
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
print(dist_LA.shape)
dist_LA.fit_transform(X[:, -2:])
coord_LA = (34.0522, -118.2437)
dist_LA = DistFromCity(coord_LA)
dist_LA.fit_transform(X[:, -2:])
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# get data
data = fetch_california_housing()
X = data['data']
y = data['target']

print(data['DESCR'])
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
from sklearn.linear_model import LinearRegression

# create model and train/fit
model = LinearRegression()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("shape of the  prediction array: {}".format(y_pred.shape))
print("shape of the training set: {}".format(X.shape))
print("β_0: {}".format(model.intercept_))

for i in range(8):
    print("β_{}: {}".format(i+1, model.coef_[i]))
print("R^2: {:g}".format(model.score(X, y)))
from sklearn.ensemble import GradientBoostingRegressor

# create model and train/fit
model = GradientBoostingRegressor()
model.fit(X, y)

# predict label values on X
y_pred = model.predict(X)

print(y_pred)
print("R^2: {:g}".format(model.score(X, y)))
from sklearn.preprocessing import StandardScaler

# create and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# scale data set
Xt = scaler.transform(X)

# create data frame with results
stats = np.vstack((X.mean(axis=0), X.var(axis=0), Xt.mean(axis=0), Xt.var(axis=0))).T
feature_names = data['feature_names']
columns = ['unscaled mean', 'unscaled variance', 'scaled mean', 'scaled variance']

df = pd.DataFrame(stats, index=feature_names, columns=columns)
df
from sklearn.compose import ColumnTransformer

col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('scaler', StandardScaler(), slice(0,6)) # first 6 columns
    ]
)

col_transformer.fit(X)
Xt = col_transformer.transform(X)

print('MedInc mean before transformation?', X.mean(axis=0)[0])
print('MedInc mean after transformation?', Xt.mean(axis=0)[0], '\n')

print('Longitude mean before transformation?', X.mean(axis=0)[-1])
print('Longitude mean after transformation?', Xt.mean(axis=0)[-1])
col_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('remove', 'drop', 0),
        ('scaler', StandardScaler(), slice(1,6))
    ]
)

Xt = col_transformer.fit_transform(X)

print('Number of features in X:', X.shape[1])
print('Number of features Xt:', Xt.shape[1])
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# construct pipeline
scaler = StandardScaler()
poly_features = PolynomialFeatures(degree=2)
lin_reg = LinearRegression()

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
pipe.named_steps
# fit/train model and predict labels
pipe.fit(X, y)
y_pred = pipe.predict(X)

print(y_pred)
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

scaler = StandardScaler()
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)

pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)

print("number of columns/features in the original data set: {}".format(X.shape[-1]))
print("number of columns/features in the new data set: {}".format(union.transform(X).shape[-1]))
print("R^2: {}".format(pipe.score(X, y)))
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
replacer = OutlierReplacer(5, 95)
replacer.fit(X) 
replacer = OutlierReplacer(5, 95)
replacer_copy = replacer.fit(X) 

print(replacer is replacer_copy)
print(id(replacer) == id(replacer_copy))
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper
    
    def fit(self, X, y=None):
        self.upper = np.percentile(X, self.q_upper, axis=0)
        self.lower = np.percentile(X, self.q_lower, axis=0)
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        # return all the indecies which are lower than lower bound
        ind_lower = X < self.lower
        # return all the indecies which are greater than upper bound
        ind_upper = X > self.upper
        
        for i in range(X.shape[-1]):
            Xt[ind_lower[:, i], i] = self.lower[i]
            Xt[ind_upper[:, i], i] = self.upper[i]
        
        return Xt
# create and fit a transformer object and transform the data
replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt = replacer.transform(X)

# plot histogram of feature 0
_, bins, _ = plt.hist(X[:, 0], density=True, bins=40, alpha=0.25, color='b')
plt.hist(Xt[:, 0], bins=bins, density=True, alpha=0.25, color='r')
plt.legend(['original', 'transformed']);
from sklearn.base import RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y_mean = np.mean(y)
        
        return self
    
    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])
mean_regressor = MeanRegressor()
mean_regressor.fit(X, y)

print(mean_regressor.predict(X))
print("R^2: {}".format(mean_regressor.score(X, y)))
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
X.shape
DropColumns([0, 1, 2, 3, 4, 5])
DropColumns([6, 7])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([6, 7])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
DropColumns([6, 7])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
print(dist_LA)
dist_SF = DistFromCity(coord_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)
print(dist_SF)
drop = DropColumns([0, 1, 2, 3, 4, 5])
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
#pipe.fit(X, y)
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])
print(drop)
union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])

union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
# pipe.fit(X, y)
from sklearn.datasets import load_wine

data = load_wine()
data
from sklearn.datasets import load_wine

data = load_wine()
X = data['data']
y = data['target']
X.shape
y
from Collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        print(c)
        return self
cls = MajorityClassifier()
cls.fit(X, y)
c
c = Counter(y)
c
c = Counter(y)
c.most_common(1)
c = Counter(y)
c.most_common(1)[0]
c = Counter(y)
c.most_common(1)[0][0]
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)
mc.mode_
cls.mode_
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
X.shape
X.shape[0]
np.ones(X.shape[0], 1)
np.ones((X.shape[0]), 1)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.ones((X.shape[0]), 1)
np.ones(3, 1)
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0]), 1), self.fraction.reshape(1, 3))
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)
cls.predict_proba(X)
np.ones((X.shape[0], 1))
cls.predict_proba(X)
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0], 1)), self.fraction.reshape(1, 3))
cls.predict_proba(X)
from sklearn.base import ClassifierMixin
from collections import Counter

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        c = Counter(y)
        self.mode_ = c.most_common(1)[0][0]
        self.fraction = np.array(list(c.values())) / X.shape[0]
        
        return self
    
    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])
    
    def predict_proba(self, X):
        return np.dot(np.ones((X.shape[0], 1)), self.fraction.reshape(1, 3))
cls = MajorityClassifier()
cls.fit(X, y)
cls.predict(X)
cls.mode_
cls.predict_proba(X)
class DistFromCity(BaseEstimator, TransformerMixin):
    def __init__(self, coord): 
        self.coord = coord
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lat = X[:, 0]
        lon = X[:, 1]
        
        dist = np.sqrt((lat - self.coord[0])**2 + (lon - self.coord[1])**2)
        
        return dist
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, ind_cols):
        self.ind_cols = ind_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.ind_cols]
coord_LA = (34.0522, -118.2437)
coord_SF = (37.7749, -122.4194)
dist_LA = DistFromCity(coord_LA)
dist_SF = DistFromCity(coord_SF)

drop = DropColumns([0, 1, 2, 3, 4, 5])

union = FeatureUnion([
    ('drop', drop), 
    ('LA', dist_LA), 
    ('SF', dist_SF)
])

pipe = Pipeline([
    ('union', union), 
    ('regressor', LinearRegression())
])
pipe.fit(X, y)
# pipe.fit(X, y)
cls.predict_proba(X)[0]
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
data = fetch_california_housing()
X = data['data']
y = data['target']
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('rgr', LinearSVR())
])
pipe.fit(X, y)
print(pipe.score(X, y))
data = fetch_california_housing()
X = data['data']
y = data['target']
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('rgr', LinearSVR())
])
pipe.fit(X, y)
print(pipe.score(X, y))
import dill
with open('my_model.dill', 'wb') as f:
    dill.dump(pipe, f)
!ls -alh my_model.dill
# restore from external dill file
with open('my_model.dill', 'rb') as f:
    pipe = dill.load(f)
pipe.score(X, y)
# if we need to compress the data
import gzip 
with gzip.open('my_model.dill.gz', 'wb') as f:
    dill.dump(pipe, f)
!ls -alh my_model.dill.gz
%logstop
%logstart -rtq ~/.logs/ML_Scikit_Learn.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
