from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10




#read the file
df = pd.read_excel(r'C:\Users\admin\Downloads\interview prep\citi\train.xlsx')
df.isnull().sum()
#print the head
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
df.columns
#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

#check stationarity
#define function for ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
#apply adf test on the series
adf_test(df['Close'])

#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
#apply adf test on the series
kpss_test(df['Close'])

from statsmodels.tsa.stattools import adfuller
#series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
#X = df[~4377:]['1D']
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))

df['1D']=df['Close'].diff()
adf_test(df[~4377:]['1D'])
kpss_test(df[~4377:]['1D'])

plt.plot(df[~4377:]['1D'], label='Close Price history')
plt.plot(df[~4377:]['1D'].rolling(5).mean(), color='red', label='Rolling Mean')
plt.plot(df[~4377:]['1D'].rolling(5).std(), color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')



#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['1D'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
new_data.dropna(inplace=True)

#creating train and test sets
dataset = new_data.values
tran=new_data['2006-01-04 00:00:00':'2010-12-31 00:00:00']
train = tran.values
val= new_data['2011-01-01 00:00:00':'2017-12-29 00:00:00']
valid = val.values
#print(dataset)

#converting dataset into x_train and y_train
#feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
#print(scaled_data)
#creating adata structre with 60 time stamps and 1 output.
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
#print(x_train)
#print(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
#initializing the RNN
model = Sequential()
#adding the first LSTM layer and some dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
#adding a second LSTM layer 
#can add more LSTM layers and add droupout rates as well
model.add(LSTM(units=50))#50 neurons
#adding the output layer
model.add(Dense(units=1))#to give me just 1 output
#compiling the rnn
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#fitting the RNN to the training set
model.fit(x_train, y_train, epochs=1, batch_size=25, verbose=1)

#predicting test values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
print(inputs)
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
mean_absolute_error(valid,closing_price)
mean_absolute_error(val['actual'],val['NSPredictions'])
accuracy_score(val['actual'],val['NSPredictions'])
model.score()
len(valid)
len(closing_price)
#for plotting
val['Predictions'] = closing_price
val['actual']=data['Close']['2011-01-01 00:00:00':'2017-12-29 00:00:00']
ts = pd.to_datetime("2010-12-31 00:00:00", format="%Y-%m-%d")
new_row = pd.DataFrame([[1.4114337669603358,0,  41.85590339229579]], columns = ["Close", 'Predictions', 'actual'], index=[ts])
val = pd.concat([val, pd.DataFrame(new_row)], ignore_index=False)


#val.loc['2010-12-31 00:00:00'] = [1.4114337669603358, 0, 41.85590339229579]  # adding a row
  
val = val.sort_index()
#x, x_diff = val['actual'].loc['2010-12-31 00:00:00'], val['Predictions'].loc['2011-01-01 00:00:00':]
#val['NSPredictions'] = np.r_[x, x_diff].cumsum()
#inverting the 1st differencing
val.index=range(0,len(val))
for i in range(0,len(val)):
    val['NSPredictions'][i+1]=val['actual'][i]+val['Predictions'][i+1]


plt.plot(tran['Close'])
plt.plot(val[['actual','NSPredictions']])
#accuracy_score(val['Close'],val['Predictions'])
#val.to_csv(r'C:\Users\admin\Downloads\interview prep\citi\predictions.csv')
#conda install ipynb-py-convert
#set(PYTHONUTF8=1)
#ipynb-py-convert 'untitled1.py' 'untitled1.ipynb'
#pip install spyder-notebook
