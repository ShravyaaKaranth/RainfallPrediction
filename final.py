# Import the required libraries
from tkinter import *

# Create an instance of tkinter frame
win = Tk()

# Define the size of window or frame
win.geometry("715x250")

# Set the Menu initially
output = StringVar()
output.set("Select State")

with open('states.txt') as file:
    options = [line.rstrip() for line in file]



# def detect_state(*args):
#     print(
#         output.get()
#     )  
def detect_state(*args):
    
    global x 
    x = output.get()
    with open ('gfg.txt', 'w') as file:
        file.write(x)


# Create a dropdown Menu
drop = OptionMenu(win, output, *options)
drop.pack()
output.trace("w", detect_state)
button = Button(win, text="Predict").pack()
win.after(5000,lambda:win.destroy())
win.mainloop()

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

#reading te data
data=pd.read_csv("India_Rainfall.csv",engine="python")

data.corr()

#input = pd.read_csv("gfg.txt", sep="\n", header=None)

da = data['State'] == x
df_new = pd.DataFrame(data[da])
# daa = df_n['District'] == input[0][1]
# df_new = pd.DataFrame(df_n[daa])

df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
df_new.dropna(inplace=True)

def norm(df_name, column_name):
  p_list = df_name[column_name].tolist()
  norm_p= []
  for i in df_name[column_name].values:
    z = (i - min(p_list))/ (max(p_list) - min(p_list))
    norm_p.append(z)
  # print(min(p_list))
  # print(max(p_list))
  return norm_p

Feb_norm = norm(df_new, 'Feb')
Apr_norm = norm(df_new, 'Apr')
Jun_norm = norm(df_new, 'Jun')
Jul_norm = norm(df_new, 'Jul')
Aug_norm = norm(df_new, 'Aug')
Sep_norm = norm(df_new, 'Sep')
Oct_norm = norm(df_new, 'Oct')
Nov_norm = norm(df_new, 'Nov')
prcp_norm = norm(df_new, 'prcp')

anan = pd.DataFrame()

anan['Feb'] = Feb_norm
anan['Apr'] = Apr_norm
anan['Jun'] = Jun_norm
anan['Jul'] = Jul_norm
anan['Aug'] = Aug_norm
anan['Sep'] = Sep_norm
anan['Oct'] = Oct_norm
anan['Nov'] = Nov_norm
anan['prcp'] = prcp_norm

anan.head()

from sklearn.model_selection import train_test_split
y= anan['prcp']
x = anan.loc[:, ('Feb', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov')]


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, 
                                                    random_state=0)

model = MLPRegressor()
model.fit(x_train, y_train)

expected_y  = y_test
predicted_y = model.predict(x_test)


from sklearn import metrics
from sklearn.metrics import mean_squared_error
rtwo = metrics.r2_score(expected_y, predicted_y)#r2 value
rootmean =np.sqrt(mean_squared_error(predicted_y,expected_y))#rmse
print("R Squared Value " + str(rtwo))
print("Root Mean Square value " +str(rootmean))

#denormalization
new_pred = []
old_max = 1
old_min = 0
oldrange = (old_max - old_min)
new_max=200
new_min = 0
newrange = (new_max - new_min)

for i in predicted_y:
  n = (i-old_min)*newrange/oldrange+new_min
  new_pred.append(n)

