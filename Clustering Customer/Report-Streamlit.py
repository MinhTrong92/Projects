import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


#icon https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title='Cluster Customer',
                    page_icon=':bar_chart:',
                    layout='wide')
st.title('Cluster Customer')

st.sidebar.header(":question:Questions")

st.sidebar.write('1. What is the impact of our website traffic on revenue?')
st.sidebar.write('2. Which products get us pageviews and revenue?')
st.sidebar.write('3. What customer segments are there?')


# Import và xử lý dữ liệu
customer_info = pd.read_excel("F:\\Study\\X-DATA\\KHÓA 4\\Lesson 8\\DA_TEST_2\\Clustering Customer\\customers (UK).xlsx",sheet_name=0)
items = pd.read_excel('F:\\Study\\X-DATA\\KHÓA 4\\Lesson 8\\DA_TEST_2\\Clustering Customer\\customers (UK).xlsx',sheet_name=1)
customer_trans = pd.read_excel('F:\\Study\\X-DATA\\KHÓA 4\\Lesson 8\\DA_TEST_2\\Clustering Customer\\customers (UK).xlsx',sheet_name=2)

traffic_01 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_01.xlsx',sheet_name=0)
traffic_02 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_02.xlsx',sheet_name=0)
traffic_03 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_03.xlsx',sheet_name=0)
traffic_04 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_04.xlsx',sheet_name=0)
traffic_05 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_05.xlsx',sheet_name=0)
traffic_06 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_06.xlsx',sheet_name=0)
traffic_07 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_07.xlsx',sheet_name=0)
traffic_08 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_08.xlsx',sheet_name=0)
traffic_09 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_09.xlsx',sheet_name=0)
traffic_10 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_10.xlsx',sheet_name=0)
traffic_11 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_11.xlsx',sheet_name=0)
traffic_12 = pd.read_excel(r'F:\Study\X-DATA\KHÓA 4\Lesson 8\DA_TEST_2\Clustering Customer\Traffic\2020_12.xlsx',sheet_name=0)


traffic = pd.concat([traffic_01, traffic_02,traffic_03,traffic_04,traffic_05,\
                     traffic_06,traffic_07,traffic_08,traffic_09,traffic_10,traffic_11,traffic_12],\
                    ignore_index=True)


# Tách Product trong Page URL
traffic['Product'] = range(len(traffic['Page URL']))
for i in range(len(traffic['Page URL'])):
    traffic['Product'][i] = traffic['Page URL'].iloc[i].split('/')[-1]

traffic.sort_values('Posted On (DD/MM/YYYY)', ascending=True, inplace=True, ignore_index=True)

#merge data
data = customer_trans.merge(customer_info,how='inner',left_on='CustomerID', right_on='ID').\
                    merge(items,how='inner',left_on='ItemID', right_on='ItemID')


data.drop('ID', inplace=True, axis=1)

# Question 1
st.header('1. What is the impact of our website traffic on revenue?')

import matplotlib.pyplot as plt
import seaborn as sns

st.write(':flags: **Khảo sát các trường users, uniquePageviews, pageviews**')

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(traffic['Posted On (DD/MM/YYYY)'],traffic['pageviews'],)


    # thêm title, label các trục và source:
    plt.suptitle('Pageviews by Date', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    plt.xlabel('Date', size=12)
    plt.ylabel('pageviews', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(traffic['Posted On (DD/MM/YYYY)'],traffic['uniquePageviews'],)


    # thêm title, label các trục và source:
    plt.suptitle('UniquePageviews by Date', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    plt.xlabel('Date', size=12)
    plt.ylabel('uniquePageviews', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(traffic['Posted On (DD/MM/YYYY)'],traffic['users'],)


    # thêm title, label các trục và source:
    plt.suptitle('Users by Date', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    plt.xlabel('Date', size=12)
    plt.ylabel('users', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)

st.write(":point_right: **Kết luận:** Số lượng users, uniquePageviews, pageviews ổn định, không có xu hướng, có một vài thời điểm có lượng cao đột biến.")
st.write('\n')
st.write('\n')
st.write('\n')





st.write(':flags: **Compare the number of orders, items and revenue between 2 channels: In Store and Online**')
summary_trans = data.groupby(["TransactionDate","Channel"]).\
                            agg(total_orders = ('OrderID','nunique'),total_items = ('ItemID','size'), total_revenue = ('SellPrice','sum')).\
                            reset_index()


col1, col2, col3 = st.columns(3)


#plot Total Orders by Date
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(summary_trans[summary_trans["Channel"] == "In Store"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "In Store"].total_orders, \
       label="In Store")

ax.plot(summary_trans[summary_trans["Channel"] == "Online"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "Online"].total_orders, \
       label="Online")

plt.legend()
# thêm title, label các trục và source:
plt.title('Total Orders by Date', style='italic', size=12, loc='left')
plt.xlabel('Transaction Date', size=12)
plt.ylabel('Total Orders', size=12)
#plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
col1.pyplot(fig)

col1.write(':bulb: **Không có sự khác biệt** nhiều về số lượng đơn hàng bán trong ngày thông qua hai kênh **Online** và **In Store**')



#plot Total Items by Date
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(summary_trans[summary_trans["Channel"] == "In Store"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "In Store"].total_items, \
       label="In Store")

ax.plot(summary_trans[summary_trans["Channel"] == "Online"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "Online"].total_items, \
       label="Online")

plt.legend()
# thêm title, label các trục và source:
plt.title('Total Items by Date', style='italic', size=12, loc='left')
plt.xlabel('Transaction Date', size=12)
plt.ylabel('Total items', size=12)
#plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
col2.pyplot(fig)

col2.write(':bulb: **Không có sự khác biệt** nhiều về số lượng hàng bán trong ngày thông qua hai kênh **Online** và **In Store**')



#plot Total Revenue by Date
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(summary_trans[summary_trans["Channel"] == "In Store"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "In Store"].total_revenue, \
       label="In Store")

ax.plot(summary_trans[summary_trans["Channel"] == "Online"].TransactionDate,\
        summary_trans[summary_trans["Channel"] == "Online"].total_revenue, \
       label="Online")

plt.legend(loc='upper left')
# thêm title, label các trục và source:
plt.title('Total Revenue by Date', style='italic', size=12, loc='left')
plt.xlabel('Transaction Date', size=12)
plt.ylabel('Total Revenue', size=12)
#plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
col3.pyplot(fig)

col3.write(':bulb: **Không có sự khác biệt** nhiều về doanh số trong ngày thông qua hai kênh **Online** và **In Store**')


revenue = data.groupby(["TransactionDate"], as_index=False)\
                        ["SellPrice"]\
                        .agg([np.sum]).reset_index().rename(columns={'sum':'total_revenue'})

from datetime import datetime as dt
traffic['Date'] = traffic['Posted On (DD/MM/YYYY)'].dt.date

summary_traffic = traffic.groupby(["Date"]).\
                            agg(total_users = ('users','sum'), total_uniquePageviews = ('uniquePageviews','sum'), total_pageviews = ('pageviews','sum')).\
                            reset_index()

summary_traffic['Date'] = pd.to_datetime(summary_traffic['Date'])

revenue_x_traffic = revenue.merge(summary_traffic,how='inner',left_on='TransactionDate', right_on='Date')

revenue_x_traffic.drop('Date', inplace=True, axis=1)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')





#total_revenue and total_users, total_uniquePageviews, total_pageviews
st.write(':flags: **Relationship between total_revenue and total_users, total_uniquePageviews, total_pageviews**')

import scipy.stats as stats

col1, col2, col3 = st.columns(3)


#plot total_revenue & total_users
col1.write('**total_revenue and total_users**')

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(revenue_x_traffic['total_revenue'],revenue_x_traffic['total_users'])


spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False)

plt.xlabel('Total revenue', size=12)
plt.ylabel('Total users', size=12)

plt.show()
col1.pyplot(fig)

pearson_coef, p_value = stats.pearsonr(revenue_x_traffic['total_revenue'], revenue_x_traffic['total_users'])
col1.write(":arrow_right: **Correlation Coefficient** là " + str(pearson_coef.round(3)) + " và " +
    "Giá trị **P-value** là " + str(p_value.round(3)))
col1.write(":arrow_right: **Không** có mối quan hệ nào giữa **total_revenue** và **total_users**")


#plot total_revenue & total_uniquePageviews

col2.write('**total_revenue and total_uniquePageviews**')
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(revenue_x_traffic['total_revenue'],revenue_x_traffic['total_uniquePageviews'])

spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
plt.xlabel('Total revenue', size=12)
plt.ylabel('Total uniquePageviews', size=12)
#plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
col2.pyplot(fig)

pearson_coef, p_value = stats.pearsonr(revenue_x_traffic['total_revenue'], revenue_x_traffic['total_users'])
col2.write(":arrow_right: **Correlation Coefficient** là " + str(pearson_coef.round(3)) + " và " +
    "Giá trị **P-value** là " + str(p_value.round(3)))
col2.write(":arrow_right: **Không** có mối quan hệ nào giữa **total_revenue** và **total_uniquePageviews**")



#plot total_revenue & total_pageviews

col3.write('**total_revenue and total_pageviews**')
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(revenue_x_traffic['total_revenue'],revenue_x_traffic['total_pageviews'])

spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
plt.xlabel('Total revenue', size=12)
plt.ylabel('Total pageviews', size=12)
#plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

plt.show()
col3.pyplot(fig)

pearson_coef, p_value = stats.pearsonr(revenue_x_traffic['total_revenue'], revenue_x_traffic['total_users'])
col3.write(":arrow_right: **Correlation Coefficient** là " + str(pearson_coef.round(3)) + " và " +
    "Giá trị **P-value** là " + str(p_value.round(3)))
col3.write(":arrow_right: **Không** có mối quan hệ nào giữa **total_revenue** và **total_pageviews**")

st.write('\n')
st.write('\n')
st.write(':point_right: **Kết Luận**: Không có yếu tố nào trong dữ liệu website traffic tác động đến revenue')

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')

# Question 2
st.header('2. Which products get us pageviews and revenue?')

col1, col2 = st.columns(2)

with col1:
    st.write('**Top 10 sản phẩm mang lại pageviews cao nhất**')

    pageviews = traffic.groupby(["Brand","Product"]).agg(total_pageviews = ('pageviews','sum')).reset_index()

    pageviews.sort_values('total_pageviews', ascending=False, inplace=True, ignore_index=True)

    top10_views = pageviews.head(10).copy()

    top10_views.sort_values('total_pageviews', ascending=True, inplace=True, ignore_index=True)

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bar1 = ax.barh(top10_views['Product'],top10_views['total_pageviews'],data=top10_views,color='#87ceeb')

    # Show dữ liệu
    i = 0
    for p in bar1.patches:
        t1 = ax.annotate(top10_views.total_pageviews[i], xy=(p.get_width()+200, p.get_y()+p.get_height()/3))
        t1.set(size=8)
        i+=1

    # bỏ đường kẻ ở trên và bên phải chart
    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False)

    # bỏ dấu tích ở 2 trục
    ax.tick_params(left=False, bottom=False)

    plt.suptitle('The number of views by Product', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    plt.xlabel('The number of views', size=12)
    plt.ylabel('Product', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()

    st.pyplot(fig)



with col2:
    st.write('**Top 10 sản phẩm mang lại revenue cao nhất**')
    revenue = data.groupby(["Brand","Product"]).agg(total_revenue = ('SellPrice','sum')).reset_index()

    revenue.sort_values('total_revenue', ascending=False, inplace=True, ignore_index=True)

    top10_revenue = revenue.head(10).copy()

    top10_revenue.sort_values('total_revenue', ascending=True, inplace=True, ignore_index=True)

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bar1 = ax.barh(top10_revenue['Product'],top10_revenue['total_revenue'],data=top10_revenue,color='#87ceeb')

    # Show dữ liệu
    i = 0
    for p in bar1.patches:
        t1 = ax.annotate(top10_revenue.total_revenue[i], xy=(p.get_width()+200, p.get_y()+p.get_height()/3))
        t1.set(size=8)
        i+=1
    
    # bỏ đường kẻ ở trên và bên phải chart
    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False) 
    
    # bỏ dấu tích ở 2 trục
    ax.tick_params(left=False, bottom=False) 

    # thêm title, label các trục và source:
    plt.suptitle('Revenue by Product', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    #plt.title('2nd title', style='italic', size=12, loc='left')
    plt.xlabel('Revenue', size=12)
    #plt.ylabel('Product', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)

st.write('\n')
st.write('\n')

col1, col2 = st.columns(2)

with col1:
    st.write('**Top 10 sản phẩm mang lại pageviews thấp nhất**')

    top10_views_lowest = pageviews.tail(10).copy().reset_index()

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bar1 = ax.barh(top10_views_lowest['Product'],top10_views_lowest['total_pageviews'],data=top10_views_lowest,color='#87ceeb')

    # Show dữ liệu
    i = 0
    for p in bar1.patches:
        t1 = ax.annotate(top10_views_lowest.total_pageviews[i], xy=(p.get_width()+0.01, p.get_y()+p.get_height()/3))
        t1.set(size=8)
        i+=1
        
    # bỏ đường kẻ ở trên và bên phải chart
    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False) 
        
    # bỏ dấu tích ở 2 trục
    ax.tick_params(left=False, bottom=False) 

    # thêm title, label các trục và source:
    plt.suptitle('The number of views by Product', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    #plt.title('2nd title', style='italic', size=12, loc='left')
    plt.xlabel('The number of views', size=12)
    plt.ylabel('Product', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)


with col2:
    st.write('**Top 10 sản phẩm mang lại revenue thấp nhất**')

    top10_revenue_lowest = revenue.tail(10).copy().reset_index()

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bar1 = ax.barh(top10_revenue_lowest['Product'],top10_revenue_lowest['total_revenue'],data=top10_revenue_lowest,color='#87ceeb')

    # Show dữ liệu
    i = 0
    for p in bar1.patches:
        t1 = ax.annotate(top10_revenue_lowest.total_revenue[i], xy=(p.get_width()+0.5, p.get_y()+p.get_height()/3))
        t1.set(size=8)
        i+=1
        
    # bỏ đường kẻ ở trên và bên phải chart
    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False) 
        
    # bỏ dấu tích ở 2 trục
    ax.tick_params(left=False, bottom=False) 

    # thêm title, label các trục và source:
    plt.suptitle('Revenue by Product', fontweight='bold', size=14, horizontalalignment='left', x=0.125, y = 1)
    #plt.title('2nd title', style='italic', size=12, loc='left')
    plt.xlabel('Revenue', size=12)
    plt.ylabel('Product', size=12)
    #plt.text(0.7, 0.01, "Van Minh Trong - Analytics Team", style='italic',transform=plt.gcf().transFigure)

    plt.show()
    st.pyplot(fig)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')



# Question 3
st.header('3. What customer segments are there?')

#Set data
st.write(':heavy_check_mark: Bảng tổng hợp thông tin khách hàng')

customer_data = data.groupby(["CustomerID", "FirstName", "LastName","Country","Birthday","DateJoined","Newsletter"]).\
                            agg(total_expenditures = ('SellPrice','sum')).\
                            reset_index()

from datetime import date

today = date.today()

# tính tuổi
customer_data['Age'] = range(len(customer_data['Birthday']))
for i in range(len(customer_data['Birthday'])):
    customer_data['Age'][i] = today.year - customer_data['Birthday'][i].year

# tính số năm trở thành thành viên
customer_data['Loyalty'] = range(len(customer_data['DateJoined']))
for i in range(len(customer_data['DateJoined'])):
    customer_data['Loyalty'][i] = today.year - customer_data['DateJoined'][i].year


customer = customer_data.merge(customer_info[['ID','Gender']],how='inner',left_on='CustomerID', right_on='ID')

customer.drop('ID', inplace=True, axis=1)

st.dataframe(customer)

st.write('\n')
st.write('\n')

st.write(':heavy_check_mark: Thống kê các giá trị unique trong các trường')


col1, col2, col3, col4 = st.columns(4)
col_=[col1, col2, col3, col4]
list_=['Country','Newsletter','Loyalty','Gender']
for i,j in zip(col_, list_):
    i.write("Column " + j)
    i.write(customer[j].unique())

st.write(":zap: Các trường Country, Newsletter, Loyalty, Gender chứa hữu hạn các giá trị phân biệt, không thích hợp để phân loại khách hàng")
st.write(":zap: Các trường CustomerID, FirstName, LastName, Birthday, DateJoined không chứa nhiều ý nghĩa để phân loại khách hàng")
st.write(":zap: Chọn 2 chỉ tiêu là **total_expenditures** và **Age** để phân loại phân khúc khách hàng")


#Build Model
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

st.write('\n')
st.write('\n')

st.write(':heavy_check_mark: Xây dựng mô hình')
st.write(" Sử dụng phương pháp **K-Means** để phân nhóm phân khúc khách hàng")
st.write(':one: Xác định số cụm tối ưu để phân loại khách hàng')

data_set = customer.iloc[:,[7,8]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_set)

kmeans_kwargs = {
                "init": "random",
                "n_init": 10,
                "max_iter": 300,
                "random_state": 42,
                }


col1, col2 = st.columns(2)

with col1:
    st.write("**Elbow Test**")

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    #plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(1, 11), sse)

    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False) 
    
    
    ax.xaxis.grid(linestyle='dashed')
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    st.pyplot(fig)

with col2:
    st.write("**Silhouette Coefficient**")
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

    #plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(2, 11), silhouette_coefficients)

    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False) 
        
        
    ax.xaxis.grid(linestyle='dashed')
    plt.xticks(range(2, 11))

    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    st.pyplot(fig)

st.write(":arrow_right: Qua kết quả của 2 bài test, số cụm để phân nhóm khách hàng phù hợp là 4")

st.write('\n')
st.write('\n')

st.write(":two: Chạy mô hình với số cụm để phân loại là 4")


kmeans = KMeans(n_clusters=4,**kmeans_kwargs)
kmeans.fit(scaled_features)

data_set['Cluster_Labels'] = kmeans.labels_

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(data_set[data_set['Cluster_Labels']==0].Age, data_set[data_set['Cluster_Labels']==0].total_expenditures, label = 'Cluster 1')
ax.scatter(data_set[data_set['Cluster_Labels']==1].Age, data_set[data_set['Cluster_Labels']==1].total_expenditures, label = 'Cluster 2')
ax.scatter(data_set[data_set['Cluster_Labels']==2].Age, data_set[data_set['Cluster_Labels']==2].total_expenditures, label = 'Cluster 3')
ax.scatter(data_set[data_set['Cluster_Labels']==3].Age, data_set[data_set['Cluster_Labels']==3].total_expenditures, label = 'Cluster 4')

spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False) 
    
plt.xlabel('Age', size=12)
plt.ylabel('total_expenditures', size=12)
plt.legend()
plt.show()
st.pyplot(fig)

st.write("Số lượng khách hàng theo từng nhóm")
st.dataframe(data_set['Cluster_Labels'].value_counts())

st.write('\n')
st.write('\n')

st.write(":point_right: **Kết luận:** tập khách hàng có thể chia thành 4 nhóm")
st.write(":heavy_plus_sign: Nhóm khách hàng dưới 35 tuổi, có mức chi tiêu dưới 25000")
st.write(":heavy_plus_sign: Nhóm khách hàng dưới 35 tuổi, có mức chi tiêu trên 25000")
st.write(":heavy_plus_sign: Nhóm khách hàng trên 35 tuổi, có mức chi tiêu dưới 25000")
st.write(":heavy_plus_sign: Nhóm khách hàng trên 35 tuổi, có mức chi tiêu trên 25000")








