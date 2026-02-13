import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error

import pickle 

df = pd.read_csv("M:/SEM 8 Internship/global_web_traffic_2026.csv")

df.columns = df.columns.str.strip()

df.drop(['domain','last_crawled'], axis=1 , inplace= True)

df = pd.get_dummies(df,columns=['category', 'primary_market'])



x = df.drop('monthly_visits', axis=1)

y = df['monthly_visits']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2  , random_state=42)

model =DecisionTreeRegressor()

model.fit(x_train, y_train)


prediction = model.predict(x_test)

print(prediction[:5])

error = y_test - prediction 

# plt.figure(figsize=(10,5))
# plt.hist(error,bins=10)

# plt.title("Prediction Error DIstribution")
# plt.xlabel("Error")
# plt.ylabel("Frequency")
# plt.show()


comparison = pd.DataFrame({
    'Actual':y_test, 
    'Predicted' : prediction
})

print(comparison.head(10))

accuracy = r2_score(y_test,prediction)
print("R2 Score : ", accuracy)
print("Mean Absolute Error : ", mean_absolute_error(y_test,prediction))
print("Mean Square Error : ", mean_squared_error(y_test,prediction) )

pickle.dump(model, open('model.pkl  ','wb'))
pickle.dump(x.columns , open ('columns.pkl','wb'))


# plt.figure(figsize=(10,5))
# plt.plot(y_test.values[:20],label = "Actual")
# plt.plot(prediction[:20],label ="Predicted")

# plt.legend()
# plt.title("Actual vs Predicted Visits")
# plt.xlabel("Test Data Index ")
# plt.ylabel("Monthly visited")
# plt.show()

# num_cols = df.select_dtypes(include=['int64','float64']).columns
# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())


# cat_colmns = df.select_dtypes(include=['object']).columns
# df[cat_colmns] = df[cat_colmns].fillna("Unknown")
# print(df.duplicated().sum())

# top_domain = df.sort_values(
#     by='monthly_visits',
#     ascending=False

# ).head(5)



# plt.figure(figsize=(10,6))
# sns.barplot(x = 'domain' , y = 'monthly_visits', data =top_domain )

# plt.title ("Domain vs Monthly visits ")
# plt.xlabel("Domain")
# plt.ylabel("Monthly Visits ")
# plt.xticks(rotation = 45)
# plt.show()

# plt.figure(figsize=(8,6))

# sns.heatmap(df.corr(numeric_only=True),
#             annot=True,
#             cmap="coolwarm")
# plt.title("Correlation Heatmap")

# plt.show()
 

# cat_columns = df.select_dtypes(include=['object','string']).columns

# print(cat_columns)


# le = LabelEncoder()

# df['category'] = le.fit_transform(df['category'])

# df['primary_market'] = le.fit_transform(df['primary_market'])

# print(df['category'])
# print(df['primary_market'])
































































































































































































































































