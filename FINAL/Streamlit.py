#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np


st.title("XỬ DỤNG MÔ HÌNH ARIMA DỰ ĐOÁN VN-INDEX")

#Crawling Data
st.header("1. Thu thập dữ liệu")
st.write("Dữ liệu về VN-INDEX được thu thập từ nguồn https://vn.investing.com/indices/vn-historical-data")
st.write("Khoảng thời gian thu thập từ **01/01/2017** đến **28/02/2022.**")
st.write("Sử dụng thư viện **selenium** để tiến hành thu thập dữ liệu.")

code = '''
#Open web
driver=webdriver.Chrome()
driver.get("https://vn.investing.com/indices/vn-historical-data")
wait = WebDriverWait(driver, 200)

#Set start date - end Date
date_picker = driver.find_element(by=By.XPATH, value='//input[@class="hasDatepicker"]')
driver.execute_script("arguments[0].setAttribute('value','01/01/2017 - 28/02/2022')", date_picker)

#find activate calendar widget
date_picker_1 = driver.find_elements(by=By.ID, value='widgetFieldDateRange')
date_picker_1[0].click()
time.sleep(2)

#Apply new search value
apply_button = driver.find_elements(by=By.ID, value='applyBtn')
apply_button[0].click()

#Get Data
table = driver.find_element(by=By.CLASS_NAME, value='historicalTbl')
rows = table.find_elements(by=By.TAG_NAME, value='tr')'''
st.code(code, language='python')

# Read data
data = pd.read_csv(r'https://raw.githubusercontent.com/MinhTrong92/Final-Test/main/FINAL/prices.csv', index_col=0)

st.write("**Bảng dữ liệu thô ban đầu**")

st.dataframe(data)

st.write("Bảng dữ liệu ban đầu chưa được định dạng đúng về kiểu dữ liệu")



# Xử lý dữ liệu
st.header("2. Xử lý dữ liệu")

code = '''
data['Price'] = pd.to_numeric(data['Price'].astype(str).str.replace(',', ''), errors='coerce')
data['Date'] = pd.to_datetime(data.Date,format='%d/%m/%Y')
data = data.sort_values(by="Date")
data = data.reset_index(drop=True)
data.set_index('Date',inplace=True)'''
st.code(code, language='python')

data['Price'] = pd.to_numeric(data['Price'].astype(str).str.replace(',', ''), errors='coerce')
data['Date'] = pd.to_datetime(data.Date,format='%d/%m/%Y')
data = data.sort_values(by="Date")
data = data.reset_index(drop=True)
data.set_index('Date',inplace=True)

st.write("**Bảng dữ liệu sau khi được xử lý**")
st.dataframe(data)



# Visual Data
st.subheader("Visual Data")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data)

ax.set_ylabel('Price')
ax.set_xlabel('Year', loc='right')

fig.suptitle("VN-Index", fontsize=18)
st.pyplot(fig)



# Xây dưng mô hình
st.header("3. Xây dựng mô hình")

# Kiểm định tính dừng của chuỗi dữ liệu
st.subheader("3.1. Kiểm định tính dừng của chuỗi dữ liệu")

st.write("Sử dụng **ADF Test** để kiểm định tính dừng của chuỗi dữ liệu")

code = '''
from statsmodels.tsa.stattools import adfuller

def adfuller_test(price):
    results = dict()
    result = adfuller(price)
    labels = ['ADF Test Statistic','p-value']
    for value,label in zip(result,labels):
        results[label] = value
    return results

adfuller_test(data['Price'])'''
st.code(code, language='python')


from statsmodels.tsa.stattools import adfuller

def adfuller_test(price):
	results = dict()
	result = adfuller(price)
	labels = ['ADF Test Statistic','p-value']
	for value,label in zip(result,labels):
		results[label] = value
	return results

st.write("**Kết quả kiểm định**")
st.write(adfuller_test(data['Price']))

st.write("***Nhận xét:*** Chuỗi dự liệu là chuỗi không dừng, tiến hành đưa chuỗi dữ liệu về chuối dừng bằng cách lấy sai phân bậc 1.")

# Kiểm định tính dừng của chuỗi sai phân bậc một
st.subheader("3.2. Kiểm định tính dừng của chuỗi sai phân bậc một")

# Lấy sai phân bậc 1 cho chuỗi dữ liệu
st.write("**Lấy sai phân bậc 1 cho chuỗi dữ liệu**")
code = '''
data['Price First Difference'] = data['Price'] - data['Price'].shift(1)'''
st.code(code, language='python')

data['Price First Difference'] = data['Price'] - data['Price'].shift(1)

# visual chuỗi sai phân bậc 1
fig, ax = plt.subplots()
ax.plot(data['Price First Difference'])

ax.set_ylabel('Price First Difference')
ax.set_xlabel('Year', loc='right')

st.pyplot(fig)

st.write("**Kết quả kiểm định**")
st.write(adfuller_test(data['Price First Difference'].dropna()))

st.write("***Nhận xét:*** Sau khi lấy sai phận bậc 1, chuỗi dữ liệu đã cho là chuỗi dừng. Như vậy hệ số **d** trong mô hình **ARIMA** được xác định là **1**")


# Xác định hệ số p, q cho cho mô hình ARIMA
st.subheader("3.3. Xác định hệ số p, q cho cho mô hình ARIMA")

# ACF
st.write("**3.3.1.** Sử dụng **ACF** để tìm hệ số **q**, độ trễ của quá trình trung bình trượt **MA(q)**")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


ax1 = plot_acf(data['Price First Difference'].dropna())
st.pyplot(ax1)

st.write("**Nhận xét:** Hệ số **q** được chọn là **2** vì tại độ trễ 2 độ dài đại diện cho giá trị của hệ số tự tương quan nằm ngoài khoảng tin cậy.")

#PACF
st.write("**3.3.2.** Sử dụng **PACF** xác định hệ số **p**, hệ số bậc tự do p của quá trình tự hồi qui **AR(p)**")

ax2 = plot_pacf(data['Price First Difference'].dropna())
st.pyplot(ax2)

st.write("Nhận xét Hệ số **p** được chọn là **2** vì tại độ trễ 2 độ dài đại diện cho giá trị của hệ số tự tương quan nằm ngoài khoảng tin cậy.")

st.write("**Kết hợp giữa bậc của p và q và giá trị của d=1 ta có kịch bản ARIMA(2, 1, 2)**")

# Auto ARIMA
st.subheader("3.4. Sử dụng Auto ARIMA tìm mô hình tối ưu")


train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

code = '''
import pmdarima as pm
model = pm.auto_arima(train_data.Price, start_p=0, start_q=0, test='adf',
                           max_p=5, max_q=5, m=1,
                           start_P=0, seasonal=False,
                           d=None, D=0, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(model.aic())'''
st.code(code, language='python')

import pmdarima as pm
model = pm.auto_arima(train_data.Price, start_p=0, start_q=0, test='adf',
                           max_p=5, max_q=5, m=1,
                           start_P=0, seasonal=False,
                           d=None, D=0, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
st.write(model)
st.write(model.aic())
st.write("**Kết quả:** mô hình tốt nhất thu được là **ARIMA(2,1,0)**")




# Áp dụng mô hình dự báo
st.header("4. Dự Báo")
st.write("Dữ liệu ban đầu được chia ra 80% dùng để train và 20% dùng để test")

# Create Training and Test
train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

fig, ax = plt.subplots()

ax.set_xlabel('Year', loc='right')
ax.set_ylabel('Price')
ax.plot(train_data['Price'], 'blue', label='Training Data')
ax.plot(test_data['Price'], 'green', label='Testing Data')
fig.suptitle("VN-Index", fontsize=18)
ax.legend()
st.pyplot(fig)

#Import thư viện
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# Dự báo với mô hình ARIMA(2,1,2)
st.subheader("4.1. Dự báo với mô hình ARIMA(2,1,2)")

code = '''
# Import thư viện
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

# Prediction
train_ar = train_data['Price'].values
test_ar = test_data['Price'].values
from sklearn.metrics import mean_squared_error

history = [x for x in train_ar]
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(2,1,2)) 
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)

rmse = sqrt(mean_squared_error(test_ar, predictions))
mse = mean_squared_error(test_ar, predictions)
mae = mean_absolute_error(test_ar, predictions)
'''
st.code(code, language='python')


train_ar = train_data['Price'].values
test_ar = test_data['Price'].values
from sklearn.metrics import mean_squared_error

history = [x for x in train_ar]
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(2,1,2)) 
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)

rmse = sqrt(mean_squared_error(test_ar, predictions))
mse = mean_squared_error(test_ar, predictions)
mae = mean_absolute_error(test_ar, predictions)

st.write("**Kết quả mô hình ARIMA(2,1,2)**")
st.write(model_fit.summary())
st.write("**Testing RMSE ARIMA(2,1,2): %.3f**" % rmse)
st.write("**Testing MSE ARIMA(2,1,2): %.3f**" % mse)
st.write("**MAE ARIMA(2,1,2): %.3f**" % mae)

# Visual Test and Pridiction
fig, ax = plt.subplots()

ax.set_xlabel('Month', loc='right')
ax.set_ylabel('Price')
ax.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
ax.plot(test_data.index, test_data['Price'], color='red', label='Actual Price')

fig.suptitle("VN-Index With ARIMA(2,1,2)", fontsize=18)
ax.legend()
st.pyplot(fig)



# Dự báo với mô hình ARIMA(2,1,0)
st.subheader("4.2. Dự báo với mô hình ARIMA(2,1,0)")

train_ar = train_data['Price'].values
test_ar = test_data['Price'].values
from sklearn.metrics import mean_squared_error

history = [x for x in train_ar]
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(2,1,0)) 
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)

rmse = sqrt(mean_squared_error(test_ar, predictions))
mse = mean_squared_error(test_ar, predictions)
mae = mean_absolute_error(test_ar, predictions)

st.write("**Kết quả mô hình ARIMA(2,1,0)**")
st.write(model_fit.summary())
st.write("**Testing RMSE ARIMA(2,1,0): %.3f**" % rmse)
st.write("**Testing MSE ARIMA(2,1,0): %.3f**" % mse)
st.write("**MAE ARIMA(2,1,0): %.3f**" % mae)

# Visual Test and Pridiction
fig, ax = plt.subplots()

ax.set_xlabel('Month', loc='right')
ax.set_ylabel('Price')
ax.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
ax.plot(test_data.index, test_data['Price'], color='red', label='Actual Price')

fig.suptitle("VN-Index With ARIMA(2,1,0)", fontsize=18)
ax.legend()
st.pyplot(fig)

st.write("**Các chỉ số MAE, RMSE, MSE của mô hình ARIMA(2,1,0) tốt hơn so với mô hình ARIMA(2,1,2) nên chọn mô hình **ARIMA(2,1,0)** cho dự báo VN-Index**")


# In[ ]:




