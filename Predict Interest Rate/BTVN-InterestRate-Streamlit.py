import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO



st.title("Analytics And Predict Interest Rate")

df = pd.read_csv(r'https://raw.githubusercontent.com/MinhTrong92/Final-Test/main/Predict%20Interest%20Rate/loans_full_schema.csv')

st.write("Raw Data")

st.dataframe(df)

#Describe the dataset and any issues with it
st.header("1. Describe the dataset and any issues with it")

st.write("**Kiểm tra kích thước bộ dữ liệu**")

st.dataframe(df.shape)

st.write("Dữ liệu có **10000** dòng với **55** thuộc tính")

st.write("**Kiểm tra giá trị NA**")

st.dataframe(df.isnull().sum())
st.write("Các cột chưa nhiều giá trị NA như **annual_income_joint**, **verification_income_joint**, **debt_to_income_joint**, **months_since_last_delinq**, **months_since_90d_late** sẽ xem xét bỏ qua trong quá trình phân tích dữ liệu")

st.write("Kiểu dữ liệu cơ bản được định đúng")



st.header("2. Visualizations")

import matplotlib.pyplot as plt
import seaborn as sns

#State
st.subheader("State")

summary_by_state = df.groupby(["state"], as_index=False)\
    ["loan_amount"]\
    .agg([np.size, np.sum]).reset_index().rename(columns={'size':'no_borrowers','sum':'total_loan_amount'})


summary_by_state.sort_values('no_borrowers', ascending=False, inplace=True, ignore_index=True)

summary_by_state['Percentage'] = summary_by_state['no_borrowers']/summary_by_state['no_borrowers'].sum()*100

summary_by_state['Cummulate']  = summary_by_state['Percentage'].cumsum()

#plot

# Set figure and axis
fig, ax1 = plt.subplots(figsize=(25,15))
xs = summary_by_state['state']
ys = summary_by_state['no_borrowers']
yl = summary_by_state['Cummulate']

# Plot bars
ax1.bar(xs, ys, color = '#3cb371')

#ax1.set_title("The number of borrowers by State", fontsize = 25, color = 'blue')
ax1.set_xlabel("State", fontsize = 15)
ax1.set_ylabel("The number of borrowers", fontsize = 15)

# Second y axis (i.e. cumulative percentage)
ax2 = ax1.twinx()
ax2.plot(xs,yl, color="#00ffff", marker="o", ms=10)

ax2.axhline(80, color="blue", linestyle="--") # Tạo đường 80%

#annotate for line
for x,y in zip(xs,yl):
    if y > 80:
        break
    else:
        label = "{:.2f}".format(y) + '%'

        ax2.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,3), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
        
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.tick_params(left=False, bottom=False)
ax2.tick_params(right=False, bottom=False) 

ax2.set_ylabel("Cumulative Percentage", fontsize = 15)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)


plt.suptitle('The number of borrowers by State', fontweight='bold', size=16, horizontalalignment='left', x=0.125, y = 1)
plt.text(0.7, 0.075, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)
plt.show()
st.pyplot(fig)

st.write('**CA** là bang có số lượng người vay nhiều nhất. Các bang **TX**, **NY**, **FL** cũng có số lượng người vay vượt trội hơn so với các bang còn lại.')

st.write('**5 bang có lượng người vay cao nhất** đã chiếm gần **40%** trong tổng số lượng người vay')


#Top 5 bang có lượng tiền vay cao nhất
summary_by_state['proportion'] = summary_by_state['total_loan_amount']/summary_by_state['total_loan_amount'].sum()*100

top5_state = summary_by_state.loc[:, ['state','total_loan_amount','proportion']].head(5).copy()

top5_state['total_loan_amount'] = top5_state['total_loan_amount']/1000

top5_state.sort_values('total_loan_amount', ascending=True, inplace=True, ignore_index=True)

# plot
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.barh('state', 'total_loan_amount', data=top5_state, color='#ffe4b5')

# Show dữ liệu 
i = 0
for p in bar1.patches:
    t1 = ax.annotate(top5_state.total_loan_amount[i], xy=(p.get_width()+200, p.get_y()+p.get_height()/2)) # số 200 ở đây tùy scale của chart mà ta tinh chỉnh để số liệu tách ra xa khỏi bar cho đẹp
    t2 = ax.annotate('('+str(top5_state.proportion.round(2)[i])+'%)', xy=(p.get_width()+200, p.get_y()+0.15))
    t1.set(size=12)
    t2.set(size=12)
    i+=1
    
# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False) 

# thêm title, label các trục và source:
plt.suptitle('Top 5 states with the highest total loan amount', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
#plt.title('2nd title', style='italic', size=12, loc='left')
plt.xlabel('Total loan amount (thousand)', size=12)
plt.ylabel('State', size=12)
plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()

st.pyplot(fig)

st.write('Đi cũng với số lượng người vay cao nhất, thì 5 bang **CA**, **TX**, **NY**, **FL**, **IL** cũng có lượng tiền được cho vay cao nhất')

#Loan purpose
st.subheader("Loan purpose")

summary_by_loan_purpose = df.groupby(["loan_purpose"], as_index=False)\
    ["loan_amount"]\
    .agg([np.size, np.sum]).reset_index().rename(columns={'size':'no_borrowers','sum':'total_loan_amount'})

summary_by_loan_purpose.sort_values('no_borrowers', ascending=True, inplace=True, ignore_index=True)

# plot
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.barh('loan_purpose', 'no_borrowers', data=summary_by_loan_purpose, color='#87ceeb')

# Show dữ liệu
i = 0
for p in bar1.patches:
    t1 = ax.annotate(summary_by_loan_purpose.no_borrowers[i], xy=(p.get_width()+20, p.get_y()+p.get_height()/3)) # số 200 ở đây tùy scale của chart mà ta tinh chỉnh để số liệu tách ra xa khỏi bar cho đẹp
    t1.set(size=8)
    i+=1
    
# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False) 

# thêm title, label các trục và source:
plt.suptitle('The number of borrowers by Loan purpose', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
#plt.title('2nd title', style='italic', size=12, loc='left')
plt.xlabel('The number of borrowers', size=12)
plt.ylabel('Loan purpose', size=12)
plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
st.pyplot(fig)

st.write('Các khoản vay chủ yếu cho hoạt động hợp nhất nợ *(debt_consolidation) với số lượng hợp đồng là hơn 5000*')
st.write('Tiếp đến là số lượng các khoản vay để mua sắm thông qua các thẻ tín dụng và sửa chữa, cải tạo nhà ở cũng ở mức cao')


st.subheader("Annual income")

#plot

fig, ax = plt.subplots(figsize=(12, 6))

ax=sns.histplot(data=df, x=df.annual_income, kde=True, color='#5f9ea0')
ax.axvline(42000, color="blue", linestyle="--")
ax.axvline(75000, color="blue", linestyle="--")
plt.xlim(0, 300000)

ax.yaxis.grid(color='gray', linestyle='dashed')
# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False) 

# thêm title, label các trục và source:
plt.suptitle('Distribution of annual income', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
#plt.title('2nd title', style='italic', size=12, loc='left')
plt.xlabel('Annual income', size=12)
plt.ylabel('Count', size=12)
plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
st.pyplot(fig)

st.write('Khách hàng vay tiền có mức thu nhập hàng nằm tập trung vào khoảng **42000** đến **75000**')

#Loan Status and Debt to Income
st.subheader("Loan Status and Debt to Income")

df1 = df.loc[:, ['loan_status','debt_to_income']].copy()

median_list = df1.groupby('loan_status', as_index=False)['debt_to_income'].median().rename(columns={'debt_to_income':'median'})

median_list.sort_values('median', ascending=True, inplace=True, ignore_index=True)

df2 = pd.merge(df1,median_list,on='loan_status',how='left')

df2.sort_values('median', ascending=True, inplace=True, ignore_index=True)

#plot
fig, ax = plt.subplots(figsize=(12, 6))
box_width=0.7

ax=sns.boxplot(x=df2.loan_status, y=df2.debt_to_income, data=df2, width=box_width, color='#adff2f')

# show medians
i = 0
for i in range(median_list.shape[0]):
    x = i #+box_width/2*1.05
    y = median_list.loc[i,['median']]
    ax.annotate('%.2f' %median_list.loc[i,['median']],
                  (x,y),
                  #color='white',
                  weight='semibold',
                ha='center',
                  size=8
                 )

# Add x, y gridlines
ax.yaxis.grid(color='gray', linestyle='dashed')

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False) 

# thêm title, label các trục và source:
plt.suptitle('The relationship between loan status and debt to income', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
#plt.title('2nd title', style='italic', size=12, loc='left')
#plt.xlabel('loan_status', size=12)
#plt.ylabel('debt_to_income', size=12)
plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.ylim(0, 50)
plt.show()
st.pyplot(fig)

st.write('Những khoản vay trong tình trạng trễ từ 31-120 ngày **Late (31-120 days)** thuộc về nhóm đối tượng vay có tỷ lệ nợ trên tổng thu nhập thấp nhất, thấp hơn cả những khoản vay có trạng thái đã trả xong **Full Paid**.')

st.write('Những khoản vay trong tình trạng **Charged Off** (mất khả năng thanh toán) có tỷ lệ nợ trên tổng thu nhập rõ ràng cao hơn so với các nhóm còn lại')


#loan grade and interest rate
st.subheader("Loan Grade and Interest Rate")

df_grade = df.loc[:, ['grade','interest_rate']].copy()

df_grade.sort_values('grade', ascending=True, inplace=True, ignore_index=True)

median_list1 = df_grade.groupby('grade')['interest_rate'].median()

#plot
fig, ax = plt.subplots(figsize=(12, 6))

box_width=0.7
ax=sns.boxplot(x=df_grade.grade, y=df_grade.interest_rate, data=df_grade, width=box_width, color='#deb887')
ax.plot(range(len(median_list1)), median_list1, 'bo--', label= "versicolor")

# show medians
i = 0
for i in range(len(median_list1)):
    x = i+box_width/2*1.05
    y = median_list1[i]
    ax.annotate('%.2f' %median_list1[i],
                  (x,y-0.3),
                  #color='white',
                  weight='semibold',
                  size=8
                 )

# Add x, y gridlines
ax.yaxis.grid(color='gray', linestyle='dashed')

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False) 

# thêm title, label các trục và source:
plt.suptitle('The relationship between loan status and debt to income', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
#plt.title('2nd title', style='italic', size=12, loc='left')
#plt.xlabel('loan_status', size=12)
#plt.ylabel('debt_to_income', size=12)
plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)


plt.show()
st.pyplot(fig)

st.write('Thứ hạng đánh giá ảnh hưởng rất nhiều đến mức lãi suất mà người vay phải trả. Mức đánh giá càng cao thì mức lãi suất càng thấp, mức đánh giá càng thấp thì lãi xuất sẽ càng cao')

#build Models

import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

st.header("3. Models")
st.subheader("3.1 Mối tương quan giữa biến *interest_rate* với một số biến")

st.write('Khảo sát mội tương quan giữa biến **interest_rate** với một số biến như **loan_amount**, **term**, **grade**, **sub_grade**, **annual_income**, **homeownership**, **verified_income**, **debt_to_income**')

st.write('dữ liệu khảo sát')
data = df.loc[:,['interest_rate','loan_amount','term','grade','sub_grade','annual_income','homeownership','verified_income','debt_to_income']].copy()

st.dataframe(data)

st.write('**Hiệu chỉnh dữ liệu khảo sát**')

st.write("**Thay thế các giá trị null của trường debt_to_income bằng giá trị mean của trường debt_to_income**")

mean = data['debt_to_income'].mean()
data['debt_to_income'] = data['debt_to_income'].replace(np.nan,mean)

code = '''
mean = data['debt_to_income'].mean()
data['debt_to_income'] = data['debt_to_income'].replace(np.nan,mean)'''
st.code(code, language='python')

st.write("**LabelEncoder cho các trường grade, sub_grade, homeownership, verified_income**")

list_lable = ['grade', 'sub_grade', 'homeownership', 'verified_income']
for i in list_lable:
    data[i] = LabelEncoder().fit_transform(data[i])


code = '''
list_lable = ['grade', 'sub_grade', 'homeownership', 'verified_income']
for i in list_lable:
    data[i] = LabelEncoder().fit_transform(data[i])'''
st.code(code, language='python')

st.write("**Bảng dữ liệu sau khi được hiệu chỉnh**")

st.dataframe(data)

st.write("**Hệ số tương quan giữa các biến**")

top = 10
corr = data.corr()
top15 = corr.nlargest(top, 'interest_rate')['interest_rate'].index
corr_top15 = data[top15].corr()

fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_top15, square=True, ax=ax, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size':12})
plt.title('Top correlated features of dataset', size=16)
plt.show()

st.pyplot(fig)

st.write('Có 2 biến **grade** và **sub_grade** có hệ số tương quan đối với biến **interest_rate** là lớn so với các biến còn lại')

st.write('**Kiểm tra mối tương quan giữa interest_rate và grade**')
fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.regplot(x=data['grade'], y=data['interest_rate'], data=data)

ax.set_xticks((0,1,2,3,4,5,6))
ax.set_xticklabels(('A', 'B', 'C', 'D', 'E', 'F','G'))

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False)
plt.suptitle('The relationship between interest_rate and grade', fontweight='bold', size=12, horizontalalignment='left', x=0.125, y = 1)
plt.show()
st.pyplot(fig)

pearson_coef, p_value = stats.pearsonr(data['grade'], data['interest_rate'])
st.write("**Correlation Coefficient** giữa **grade** và **interest_rate** là **" + str(pearson_coef.round(4)) + "** và giá trị **P-value là** **" + str(p_value.round(4)) + "**")
st.write("interest_rate và grade có mối quan hệ tuyến tính đồng biến và p_value < 0.001 cho thấy mức độ tương quan của hai biến này có ý nghĩa thống kê. Có thể sử dụng biến grade để dự báo cho biến interest_rate")

st.write('**Kiểm tra mối tương quan giữa interest_rate và sub_grade**')

fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.regplot(x=data['sub_grade'], y=data['interest_rate'], data=data)

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False)
plt.suptitle('The relationship between interest_rate and sub_grade', fontweight='bold', size=12, horizontalalignment='left', x=0.125, y = 1)
plt.show()
st.pyplot(fig)

pearson_coef, p_value = stats.pearsonr(data['sub_grade'], data['interest_rate'])
st.write("**Correlation Coefficient** giữa **sub_grade** và **interest_rate** là **" + str(pearson_coef.round(4)) + "** và giá trị **P-value là** **" + str(p_value.round(4)) + "**")

st.write("interest_rate và sub_grade có mối quan hệ tuyến tính đồng biến và p_value < 0.001 cho thấy mức độ tương quan của hai biến này có ý nghĩa thống kê. Có thể sử dụng biến sub_grade để dự báo cho biến interest_rate")



#Build Models
st.subheader('3.2 Xây dựng mô hình Linear Regression')
st.write("**Import thư viện**")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

code = '''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split'''
st.code(code, language='python')


st.write("**Phân chia dữ liệu**")

X = data[['grade','sub_grade']]
y = data['interest_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

code = '''
X = data[['grade','sub_grade']]
y = data['interest_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'''
st.code(code, language='python')

st.write("**Train mô hình**")

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


code = '''
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)'''
st.code(code, language='python')

y_pred = lin_reg.predict(X_test)

st.write("Hệ số tự do : " + str(lin_reg.intercept_.round(4)))
st.write("Hệ số của biến grade và sub_grade lần lượt là: " + str(lin_reg.coef_[0].round(4)) + " và " + str(lin_reg.coef_[1].round(4)))
st.write("**Mô hình dự báo:** interest_rate = "+ str(lin_reg.intercept_.round(4)) +" + "+ str(lin_reg.coef_[0].round(4)) + "* grade " +" + "+ str(lin_reg.coef_[1].round(4)) + "* sub_grade")

st.write('The full R-square is : ' +str((lin_reg.score(X,y)*100).round(3)) + '%')
st.write('The train R-square is : ' +str((lin_reg.score(X_train, y_train)*100).round(3)) + '%')
st.write('The test R-square is : ' +str((lin_reg.score(X_test, y_test)*100).round(3)) + '%')
st.write("**Hơn 98% interest_rate có thể giải thích được bằng mô hình trên**")

#Find MSE , MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
st.write("The MSE of interest_rate and predicted value is : " + str(mse.round(3)))
st.write("The MAE of interest_rate and predicted value is : " + str(mae.round(3)))


#Predict interest_rate by Linear Regression model
st.write("**Predict interest_rate by Linear Regression model**")

col1, col2 = st.columns(2)

#image = Image.open(r'F:\Study\X-DATA\KHÓA 4\Lesson 7\Test1\hinh1.png')
col1.image(r'https://raw.githubusercontent.com/MinhTrong92/Final-Test/main/Predict%20Interest%20Rate/hinh1.png', width=300)


grade_ = col2.number_input("Insert grade")

sub_grade_ = col2.number_input("Insert sub_grade")

interest_rate_predict_linear = lin_reg.predict([[grade_,sub_grade_]])[0].round(3)

col2.write("**Predicted value of interest_rate is " + str(interest_rate_predict_linear) + "%**")


st.subheader('3.3 Xây dựng mô hình Polynomial Regression')
st.write("Thực hiện train mộ hình dữ liệu **bậc 4** với biến phụ thuộc là **interest_rate** và biến độc lập là **sub_grade**")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

dataset = data.loc[:,['interest_rate','sub_grade']]

X = np.array(dataset['sub_grade']).reshape(-1,1)
y = np.array(dataset['interest_rate']).reshape(-1,1)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

code = '''
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)'''
st.code(code, language='python')

st.write("Kết quả sau mô hình")

# Visualizing the Polymonial Regression results
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_test, y_test, color='red', label='Test')
ax.scatter(X_test, pol_reg.predict(poly_reg.fit_transform(X_test)), color='blue', label='Predict')
plt.xlabel('sub_grade')
plt.ylabel('interest_rate')

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
# bỏ dấu tích ở 2 trục
ax.tick_params(left=False, bottom=False)

ax.legend()
plt.suptitle('Polynomial Regression', fontweight='bold', size=12, horizontalalignment='left', x=0.125, y = 1)
plt.show()
st.pyplot(fig)

st.write('The full R-square is : ' +str((r2_score(y, pol_reg.predict(poly_reg.fit_transform(X)))*100).round(3)) + '%')
st.write('The train R-square is : ' +str((r2_score(y_train, pol_reg.predict(poly_reg.fit_transform(X_train)))*100).round(3)) + '%')
st.write('The test R-square is : ' +str((r2_score(y_test, pol_reg.predict(poly_reg.fit_transform(X_test)))*100).round(3)) + '%')

st.write("**Mô hình Polynomial Regression có thể giải thích được hơn 99% về interest rate**")

#Find MSE , MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, pol_reg.predict(poly_reg.fit_transform(X_test)))
mae = mean_absolute_error(y_test, pol_reg.predict(poly_reg.fit_transform(X_test)))
st.write("The MSE of interest_rate and predicted value is : " + str(mse.round(3)))
st.write("The MAE of interest_rate and predicted value is : " + str(mae.round(3)))

st.write("**Kết luận:** mô hình **Polynomial Regression** giải thích **tốt hơn** so với mô hình **Linear Regression**")



#Predict interest_rate by Polynomial Regression model
st.write("**Predict interest_rate by Polynomial Regression model**")

col1, col2 = st.columns(2)

#image = Image.open(r'https://raw.githubusercontent.com/MinhTrong92/Final-Test/main/Predict%20Interest%20Rate/hinh1.png')
col1.image(r'https://raw.githubusercontent.com/MinhTrong92/Final-Test/main/Predict%20Interest%20Rate/hinh1.png', width=220)


number = col2.number_input("Insert sub_grade to predict interest_rate")

interest_rate_predict = pol_reg.predict(poly_reg.fit_transform([[number]]))[0][0].round(4)

col2.write("**Predicted value of interest_rate is " + str(interest_rate_predict) + "%**")









