# Sun, 19 Jul 2020 20:20:01
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge# Sun, 19 Jul 2020 20:21:09
X = fetch_california_housing()['data']
y = fetch_california_housing()['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)# Sun, 19 Jul 2020 20:24:47
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', Ridge())
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))# Sun, 19 Jul 2020 20:24:59
pipe.get_params()#[Out]# {'memory': None,
#[Out]#  'steps': [('scaler',
#[Out]#    StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#   ('regressor',
#[Out]#    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#[Out]#          normalize=False, random_state=None, solver='auto', tol=0.001))],
#[Out]#  'verbose': False,
#[Out]#  'scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
#[Out]#  'regressor': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#[Out]#        normalize=False, random_state=None, solver='auto', tol=0.001),
#[Out]#  'scaler__copy': True,
#[Out]#  'scaler__with_mean': True,
#[Out]#  'scaler__with_std': True,
#[Out]#  'regressor__alpha': 1.0,
#[Out]#  'regressor__copy_X': True,
#[Out]#  'regressor__fit_intercept': True,
#[Out]#  'regressor__max_iter': None,
#[Out]#  'regressor__normalize': False,
#[Out]#  'regressor__random_state': None,
#[Out]#  'regressor__solver': 'auto',
#[Out]#  'regressor__tol': 0.001}
# Sun, 19 Jul 2020 20:26:07
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import numpy as np# Sun, 19 Jul 2020 20:27:40
param_grid = {
    "regressor__alpha": np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2, verbose=1)# Sun, 19 Jul 2020 20:27:57
param_grid = {
    "regressor__alpha": np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)#[Out]# GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#              estimator=Pipeline(memory=None,
#[Out]#                                 steps=[('scaler',
#[Out]#                                         StandardScaler(copy=True,
#[Out]#                                                        with_mean=True,
#[Out]#                                                        with_std=True)),
#[Out]#                                        ('regressor',
#[Out]#                                         Ridge(alpha=1.0, copy_X=True,
#[Out]#                                               fit_intercept=True, max_iter=None,
#[Out]#                                               normalize=False,
#[Out]#                                               random_state=None, solver='auto',
#[Out]#                                               tol=0.001))],
#[Out]#                                 verbose=False),
#[Out]#              iid='warn', n_jobs=2,
#[Out]#              param_grid={'regressor__al...3240e-03, 8.85866790e-03,
#[Out]#        1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#        3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#        6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#        1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#[Out]#              scoring=None, verbose=1)
# Sun, 19 Jul 2020 20:28:23
grid_search.best_params_#[Out]# {'regressor__alpha': 12.742749857031322}
# Sun, 19 Jul 2020 20:28:31
grid_search.best_score_#[Out]# 0.6053956962874548
# Sun, 19 Jul 2020 20:30:32
from tempfile import mkdtemp
from shutil import rmtree# Sun, 19 Jul 2020 20:32:44
cachedir = mkdtemp()
pipe_cache = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', Ridge())
], memory=cachedir)
pipe_cache.fit(X_train, y_train)#[Out]# Pipeline(memory='/tmp/tmpysv1gnen',
#[Out]#          steps=[('scaler',
#[Out]#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#                 ('regressor',
#[Out]#                  Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
#[Out]#                        max_iter=None, normalize=False, random_state=None,
#[Out]#                        solver='auto', tol=0.001))],
#[Out]#          verbose=False)
# Sun, 19 Jul 2020 21:01:43
param_grid = {
    "alpha": np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, n_jobs=2, verbose=1)

pipe2 = Pipeline([
    ('scaler', StandardScaler()), 
    ('grid_search', grid_search)
])
pipe2.fit(X_train, y_train)#[Out]# Pipeline(memory=None,
#[Out]#          steps=[('scaler',
#[Out]#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#                 ('grid_search',
#[Out]#                  GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#                               estimator=Ridge(alpha=1.0, copy_X=True,
#[Out]#                                               fit_intercept=True, max_iter=None,
#[Out]#                                               normalize=False,
#[Out]#                                               random_state=None, solver='auto',
#[Out]#                                               tol=0.001),
#[Out]#                               iid='warn', n_jobs=2,
#[Out]#                               param_grid={'alpha': array([1.00000000e-03...90e-03,
#[Out]#        1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#        3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#        6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#        1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#                               pre_dispatch='2*n_jobs', refit=True,
#[Out]#                               return_train_score=False, scoring=None,
#[Out]#                               verbose=1))],
#[Out]#          verbose=False)
# Sun, 19 Jul 2020 21:02:05
pipe2.named_steps#[Out]# {'scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
#[Out]#  'grid_search': GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#               estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
#[Out]#                               max_iter=None, normalize=False, random_state=None,
#[Out]#                               solver='auto', tol=0.001),
#[Out]#               iid='warn', n_jobs=2,
#[Out]#               param_grid={'alpha': array([1.00000000e-03, 2.06913808e-03, 4.28133240e-03, 8.85866790e-03,
#[Out]#         1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#         3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#         6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#         1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#               pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#[Out]#               scoring=None, verbose=1)}
# Sun, 19 Jul 2020 21:02:27
pipe2.named_steps['grid_search']#[Out]# GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#              estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
#[Out]#                              max_iter=None, normalize=False, random_state=None,
#[Out]#                              solver='auto', tol=0.001),
#[Out]#              iid='warn', n_jobs=2,
#[Out]#              param_grid={'alpha': array([1.00000000e-03, 2.06913808e-03, 4.28133240e-03, 8.85866790e-03,
#[Out]#        1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#        3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#        6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#        1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#[Out]#              scoring=None, verbose=1)
# Sun, 19 Jul 2020 21:02:55
pipe2.named_steps['grid_search'].best_params_#[Out]# {'alpha': 26.366508987303554}
# Sun, 19 Jul 2020 21:08:53
from sklearn.decomposition import PCA# Sun, 19 Jul 2020 21:12:04
pipe_3 = Pipeline([
    ('scaler', StandardScaler()), 
    ('dim-red', PCA()), 
    ('regressor', Ridge())
])

param_grid = {
    'dim-red__n_components': [2, 3, 4, 5, 6], 
    'regressor__alpha': np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe_3, param_grid, cv=5, n_jobs=2, verbose=1)# Sun, 19 Jul 2020 21:13:39
pipe_3 = Pipeline([
    ('scaler', StandardScaler()), 
    ('dim-red', PCA()), 
    ('regressor', Ridge())
], memory=cachedir)

param_grid = {
    'dim-red__n_components': [2, 3, 4, 5, 6], 
    'regressor__alpha': np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe_3, param_grid, cv=5, n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)
rmtree(cachedir)# Sun, 19 Jul 2020 21:14:04
grid_search.best_params_#[Out]# {'dim-red__n_components': 6, 'regressor__alpha': 26.366508987303554}
# Sun, 19 Jul 2020 21:15:49
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import numpy as np# Sun, 19 Jul 2020 21:15:49
X = fetch_california_housing()['data']
y = fetch_california_housing()['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)# Sun, 19 Jul 2020 21:15:50
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', Ridge())
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))# Sun, 19 Jul 2020 21:15:50
pipe.get_params()#[Out]# {'memory': None,
#[Out]#  'steps': [('scaler',
#[Out]#    StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#   ('regressor',
#[Out]#    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#[Out]#          normalize=False, random_state=None, solver='auto', tol=0.001))],
#[Out]#  'verbose': False,
#[Out]#  'scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
#[Out]#  'regressor': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#[Out]#        normalize=False, random_state=None, solver='auto', tol=0.001),
#[Out]#  'scaler__copy': True,
#[Out]#  'scaler__with_mean': True,
#[Out]#  'scaler__with_std': True,
#[Out]#  'regressor__alpha': 1.0,
#[Out]#  'regressor__copy_X': True,
#[Out]#  'regressor__fit_intercept': True,
#[Out]#  'regressor__max_iter': None,
#[Out]#  'regressor__normalize': False,
#[Out]#  'regressor__random_state': None,
#[Out]#  'regressor__solver': 'auto',
#[Out]#  'regressor__tol': 0.001}
# Sun, 19 Jul 2020 21:15:50
param_grid = {
    "regressor__alpha": np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)#[Out]# GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#              estimator=Pipeline(memory=None,
#[Out]#                                 steps=[('scaler',
#[Out]#                                         StandardScaler(copy=True,
#[Out]#                                                        with_mean=True,
#[Out]#                                                        with_std=True)),
#[Out]#                                        ('regressor',
#[Out]#                                         Ridge(alpha=1.0, copy_X=True,
#[Out]#                                               fit_intercept=True, max_iter=None,
#[Out]#                                               normalize=False,
#[Out]#                                               random_state=None, solver='auto',
#[Out]#                                               tol=0.001))],
#[Out]#                                 verbose=False),
#[Out]#              iid='warn', n_jobs=2,
#[Out]#              param_grid={'regressor__al...3240e-03, 8.85866790e-03,
#[Out]#        1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#        3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#        6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#        1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#[Out]#              scoring=None, verbose=1)
# Sun, 19 Jul 2020 21:15:52
grid_search.best_params_#[Out]# {'regressor__alpha': 12.742749857031322}
# Sun, 19 Jul 2020 21:15:52
grid_search.best_score_#[Out]# 0.6053956962874548
# Sun, 19 Jul 2020 21:15:52
from tempfile import mkdtemp
from shutil import rmtree# Sun, 19 Jul 2020 21:15:52
cachedir = mkdtemp()
pipe_cache = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', Ridge())
], memory=cachedir)
pipe_cache.fit(X_train, y_train)#[Out]# Pipeline(memory='/tmp/tmp9nm015m5',
#[Out]#          steps=[('scaler',
#[Out]#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#                 ('regressor',
#[Out]#                  Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
#[Out]#                        max_iter=None, normalize=False, random_state=None,
#[Out]#                        solver='auto', tol=0.001))],
#[Out]#          verbose=False)
# Sun, 19 Jul 2020 21:15:52
param_grid = {
    "alpha": np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, n_jobs=2, verbose=1)

pipe2 = Pipeline([
    ('scaler', StandardScaler()), 
    ('grid_search', grid_search)
])
pipe2.fit(X_train, y_train)#[Out]# Pipeline(memory=None,
#[Out]#          steps=[('scaler',
#[Out]#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#[Out]#                 ('grid_search',
#[Out]#                  GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#                               estimator=Ridge(alpha=1.0, copy_X=True,
#[Out]#                                               fit_intercept=True, max_iter=None,
#[Out]#                                               normalize=False,
#[Out]#                                               random_state=None, solver='auto',
#[Out]#                                               tol=0.001),
#[Out]#                               iid='warn', n_jobs=2,
#[Out]#                               param_grid={'alpha': array([1.00000000e-03...90e-03,
#[Out]#        1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#        3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#        6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#        1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#                               pre_dispatch='2*n_jobs', refit=True,
#[Out]#                               return_train_score=False, scoring=None,
#[Out]#                               verbose=1))],
#[Out]#          verbose=False)
# Sun, 19 Jul 2020 21:15:52
pipe2.named_steps#[Out]# {'scaler': StandardScaler(copy=True, with_mean=True, with_std=True),
#[Out]#  'grid_search': GridSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#               estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
#[Out]#                               max_iter=None, normalize=False, random_state=None,
#[Out]#                               solver='auto', tol=0.001),
#[Out]#               iid='warn', n_jobs=2,
#[Out]#               param_grid={'alpha': array([1.00000000e-03, 2.06913808e-03, 4.28133240e-03, 8.85866790e-03,
#[Out]#         1.83298071e-02, 3.79269019e-02, 7.84759970e-02, 1.62377674e-01,
#[Out]#         3.35981829e-01, 6.95192796e-01, 1.43844989e+00, 2.97635144e+00,
#[Out]#         6.15848211e+00, 1.27427499e+01, 2.63665090e+01, 5.45559478e+01,
#[Out]#         1.12883789e+02, 2.33572147e+02, 4.83293024e+02, 1.00000000e+03])},
#[Out]#               pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#[Out]#               scoring=None, verbose=1)}
# Sun, 19 Jul 2020 21:15:53
pipe2.named_steps['grid_search'].best_params_#[Out]# {'alpha': 26.366508987303554}
# Sun, 19 Jul 2020 21:15:53
from sklearn.decomposition import PCA# Sun, 19 Jul 2020 21:15:53
pipe_3 = Pipeline([
    ('scaler', StandardScaler()), 
    ('dim-red', PCA()), 
    ('regressor', Ridge())
], memory=cachedir)

param_grid = {
    'dim-red__n_components': [2, 3, 4, 5, 6], 
    'regressor__alpha': np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe_3, param_grid, cv=5, n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)
rmtree(cachedir)# Sun, 19 Jul 2020 21:16:01
grid_search.best_params_#[Out]# {'dim-red__n_components': 6, 'regressor__alpha': 26.366508987303554}
# Sun, 19 Jul 2020 21:32:29
from sklearn.model_selection import RandomizedSearchCV# Sun, 19 Jul 2020 21:33:09
cachedir = mkdtemp()
pipe_3 = Pipeline([
    ('scaler', StandardScaler()), 
    ('dim-red', PCA()), 
    ('regressor', Ridge())
], memory=cachedir)

param_grid = {
    'dim-red__n_components': [2, 3, 4, 5, 6], 
    'regressor__alpha': np.logspace(-3, 3, 20)
}
grid_search = GridSearchCV(pipe_3, param_grid, cv=5, n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)
rmtree(cachedir)# Sun, 19 Jul 2020 21:33:18
grid_search.best_params_#[Out]# {'dim-red__n_components': 6, 'regressor__alpha': 26.366508987303554}
# Sun, 19 Jul 2020 21:35:19
cachedir = mkdtemp()
pipe_4 = Pipeline([
    ('scaler', StandardScaler()), 
    ('dim-red', PCA()), 
    ('regressor', Ridge())
], memory=cachedir)

param_grid = {
    'dim-red__n_components': range(1, 9), 
    'regressor__alpha': np.logspace(-3, 3, 200)
}
randomized_grid_search = RandomizedSearchCV(pipe_4, param_grid, cv=5, n_jobs=2, verbose=1, n_iter=100)
randomized_grid_search.fit(X_train, y_train)#[Out]# RandomizedSearchCV(cv=5, error_score='raise-deprecating',
#[Out]#                    estimator=Pipeline(memory='/tmp/tmpuqm7_6yv',
#[Out]#                                       steps=[('scaler',
#[Out]#                                               StandardScaler(copy=True,
#[Out]#                                                              with_mean=True,
#[Out]#                                                              with_std=True)),
#[Out]#                                              ('dim-red',
#[Out]#                                               PCA(copy=True,
#[Out]#                                                   iterated_power='auto',
#[Out]#                                                   n_components=None,
#[Out]#                                                   random_state=None,
#[Out]#                                                   svd_solver='auto', tol=0.0,
#[Out]#                                                   whiten=False)),
#[Out]#                                              ('regressor',
#[Out]#                                               Ridge(alpha=1.0, copy_X=True,
#[Out]#                                                     fit_interce...
#[Out]#        2.67384162e+02, 2.86606762e+02, 3.07211300e+02, 3.29297126e+02,
#[Out]#        3.52970730e+02, 3.78346262e+02, 4.05546074e+02, 4.34701316e+02,
#[Out]#        4.65952567e+02, 4.99450512e+02, 5.35356668e+02, 5.73844165e+02,
#[Out]#        6.15098579e+02, 6.59318827e+02, 7.06718127e+02, 7.57525026e+02,
#[Out]#        8.11984499e+02, 8.70359136e+02, 9.32930403e+02, 1.00000000e+03])},
#[Out]#                    pre_dispatch='2*n_jobs', random_state=None, refit=True,
#[Out]#                    return_train_score=False, scoring=None, verbose=1)
# Sun, 19 Jul 2020 21:35:46
randomized_grid_search.score(X_test, y_test)#[Out]# 0.5941268031952003
# Sun, 19 Jul 2020 21:36:10
grid_search.score(X_test, y_test)#[Out]# 0.4782182317645793
