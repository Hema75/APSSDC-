import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import LogisticRegression;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.tree import DecisionTreeRegressor;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.neighbors import KNeighborsRegressor;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;
from lightgbm import LGBMRegressor

import tensorflow as tf
import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;
from tensorflow.keras.callbacks import EarlyStopping;

from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



tr = pd.read_csv('../input/hck-dataset/train.csv');
te = pd.read_csv('../input/hck-dataset/test.csv')

tr = tr.dropna()

tr.isnull().sum()/tr.shape[0]

gen = {'Female': 1, 'Male':2};
tr['Gender'] = tr['Gender'].map(gen);
te['Gender'] = te['Gender'].map(gen)

com = {'Service': 1, 'Product':2};
tr['Company Type'] = tr['Company Type'].map(com);
te['Company Type'] = te['Company Type'].map(com)

setup = {'No': 1, 'Yes':2};
tr['WFH Setup Available'] = tr['WFH Setup Available'].map(setup);
te['WFH Setup Available'] = te['WFH Setup Available'].map(setup)

fineTech_appData3 = tr.drop(['Burn Rate'], axis = 1) 
sns.barplot(fineTech_appData3.columns,fineTech_appData3.corrwith(tr['Burn Rate']))

tr = tr.drop(columns = ['Company Type', 'Gender'], axis =1);
te = te.drop(columns = ['Company Type', 'Gender'], axis =1)

tr = tr.drop(columns = ['Employee ID', 'Date of Joining'], axis =1);
te = te.drop(columns = ['Employee ID', 'Date of Joining'], axis =1)

m = tr.drop(columns = ['Burn Rate'], axis =1);
n = tr['Burn Rate']

m_train, m_test, n_train, n_test = train_test_split(m, n, test_size = 0.2, random_state = 51)

n_inputs = 4;
n_outputs = 1
model = Sequential()
model.add(Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics= ['accuracy'])

model.fit(m_train, n_train, epochs = 1800, batch_size = 64, verbose = 2,  validation_data = (m_test, n_test))

md_pred = model.predict(te);
md_predData = pd.DataFrame(md_pred, columns = ['Burn Rate']);
md_predData.set_index('Burn Rate').to_csv('submissiondeep2.csv')
