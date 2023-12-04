from tkinter import *
from tkinter import ttk
import tkinter.font as tkft
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg
import pandas as pd
from Model import ARIMAM, VARM, EstimateRMSE

years = [x for x in range(2023,2030)]
months = [x for x in range(1,13)]
days = [x for x in range(1,32)]


# Prediction model
ARIMAmodel, ARIMAtrain, ARIMAtest, ARIMAprediction = ARIMAM('CI_MOVIE_VIEWING_INFO_202302.csv')
VARmodel, VARtrain, VARtest, VARprediction = VARM('CI_MOVIE_VIEWING_INFO_202302.csv')

# MSE 평가 수행
ARIMARMSE,VARRMSE=EstimateRMSE(ARIMAprediction,VARprediction,ARIMAtest)

model, train, test, prediction = None,None,None,None

# Button Event Function
def Command_Radio():
    global model, train, test, prediction
    if modelName.get() == 0:
        model, train, test, prediction = ARIMAmodel, ARIMAtrain, ARIMAtest, ARIMAprediction
        print("ARIMA")
    elif modelName.get() == 1:
        model, train, test, prediction = VARmodel, VARtrain['MOVIE_ADNC_CO'], VARtest['MOVIE_ADNC_CO'], VARprediction['MOVIE_ADNC_CO']
        print("VAR")

def Command_Button():
    print("Clicked")
    global model, train, test, prediction
    print(train)
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Data', color='orange')
    plt.plot(test.index, prediction, label='Forecast', color='green')
    if modelName.get() == 0:
        plt.title("Movie Attendance Forecast - ARIMA")
    elif modelName.get() == 1:
        plt.title("Movie Attendance Forecast - VAR")
    #plt.title('Movie Attendance Forecast')
    plt.xlabel('Date')
    plt.ylabel('Movie Attendance')
    plt.legend()
    plt.show()

def Command_Calculate():
    print("Calculate button clicked")
    global model, train, test, prediction, entry_output

    # 사용자가 선택한 년도와 월 가져오기
    selected_year = int(combobox_year.get())
    selected_month = int(combobox_month.get())
    selected_day = int(combobox_days.get())

    # 날짜 형식으로 변환 (예: '2023-04-01')
    selected_date = f"{selected_year}-{selected_month:02d}-{selected_day:02d}"

    # 예측 모델 선택 및 예측 수행
    if modelName.get() == 0:  # ARIMA
        steps_to_predict = (pd.to_datetime(selected_date) - train.index[-1]).days // 7
        predicted_value = ARIMAmodel.forecast(steps=steps_to_predict)[-1]
        MSE = ARIMARMSE
    elif modelName.get() == 1:  # VAR
        steps_to_predict = (pd.to_datetime(selected_date) - train.index[-1]).days // 7
        var_predictions = VARmodel.forecast(train.values[-VARmodel.k_ar:], steps=steps_to_predict)
        predicted_value = var_predictions[-1, 0]  # 예: 'MOVIE_ADNC_CO' 컬럼의 예측값
        print(VARmodel.forecast(train.values[-VARmodel.k_ar:], steps=steps_to_predict))
        MSE = VARRMSE
    else:
        predicted_value = "Invalid model type"
        MSE=None
        

    print(f"Predicted value: {predicted_value}")

    # 예측 결과 표시
    entry_output.configure(state='normal')
    entry_output.delete(0, END)
    entry_output.insert(0, f"{predicted_value:,.0f}")
    entry_output.configure(state='readonly')
    entry_MSE.configure(state='normal')
    entry_MSE.delete(0, END)
    entry_MSE.insert(0, f"{MSE:,.0f}")
    entry_MSE.configure(state='readonly')
    

# GUI
WIN_WIDTH = 920
WIN_HEIGHT = 300

window = Tk()

modelName = IntVar()
modelName.set(-1)


window.title("Movie num of Customer prediction model")
window.geometry( str(WIN_WIDTH)+"x"+str(WIN_HEIGHT)+"+100+100" )
window.resizable(False,False)

titleFont = tkft.Font(size=15,family='Century Schoolbook')
semi_titleFont = tkft.Font(size=13,family='돋움',weight='bold')
contentFont = tkft.Font(size=12, family='돋움')


# First Line, Region selection & announce train set
label_trainDate = Label(window,text="Trainingset Date: 2019~2022    Testingset Data: 2023",font=titleFont)
label_trainDate.place(x=10,y=10)

# Second Line, show matplotlib graph by Canvas
button_graph = Button(window,width=10,height=3,text="Show Graph",font=semi_titleFont,command=Command_Button)
button_graph.place(x=550,y=10)


# Third Line, Input, date, quarter, model(ARIMA, OTHER MODEL), calculate button
labelframe_input = LabelFrame(window,text="Input",font=semi_titleFont)
labelframe_model = LabelFrame(labelframe_input,text="Model",font=contentFont)
labelframe_date = LabelFrame(labelframe_input,text="Date",font=contentFont)
combobox_year = ttk.Combobox(labelframe_date,width=10,height=10,values=years,state="readonly")
combobox_year.set("year")
combobox_month = ttk.Combobox(labelframe_date,width=5,height=10,values=months,state="readonly")
combobox_month.set("month")
combobox_days = ttk.Combobox(labelframe_date,width=5,height=10,values=days,state="readonly")
combobox_days.set("days")
radio_ARIMA = Radiobutton(labelframe_model,text="ARIMA",value=0,font=contentFont,variable=modelName,command=Command_Radio)
radio_other = Radiobutton(labelframe_model,text="VAR",value=1,font=contentFont,variable=modelName,command=Command_Radio)
button_calculate = Button(labelframe_input,width=10,height=3,text="Calculate",font=semi_titleFont,command=Command_Calculate)

combobox_year.pack(side="left",padx=5,pady=5)
combobox_month.pack(side="left",padx=5,pady=5)
combobox_days.pack(side="right",padx=5,pady=5)
labelframe_date.pack(side="left",padx=10,pady=5)
radio_ARIMA.pack(side="left",padx=5,pady=3)
radio_other.pack(side="right",padx=5,pady=3)
labelframe_model.pack(side="left",anchor='center',padx=10,pady=5)
button_calculate.pack(side="right",anchor='e',padx=10,pady=5)
labelframe_input.place(x=10,y=100)

# Fourth Line, Output, estimated customer, accuracy, '%' label
labelframe_output = LabelFrame(window,text="Output",font=semi_titleFont)
labelfram_mse = LabelFrame(labelframe_output,text="RMSE",font=contentFont)
labelframe_estimatedCustomers = LabelFrame(labelframe_output,text="Estimated number of customers",font=contentFont)
entry_output = Entry(labelframe_estimatedCustomers,state='readonly',width=15)
entry_MSE = Entry(labelfram_mse,state='readonly',width=15)

entry_output.pack(padx=5,pady=5)
labelframe_estimatedCustomers.pack(side="left",padx=10,pady=5)
entry_MSE.pack(side="left")
labelfram_mse.pack(side='right',padx=10,pady=5)
labelframe_output.place(x=10,y=200)

window.mainloop()