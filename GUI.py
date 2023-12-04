from tkinter import *
from tkinter import ttk
import tkinter.font as tkft
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg
from Model import ARIMAM, VARM
import os

print(os.getcwd())
 
# ["강원도","강원도","경기도","경상남도","경상북도","광주광역시","대구광역시","대전광역시","부산광역시","서울특별시","세종특별시","울산광역시","인천광역시","전라남도","전라북도","제주특별자치도","충청남도","충청북도"]
years = [x for x in range(2019,2025)]
months = [x for x in range(1,13)]
quarter = [str(x)+"분기" for x in range(1,5)]



# Prediction model
ARIMAmodel, ARIMAtrain, ARIMAtest, ARIMAprediction = ARIMAM('CI_MOVIE_VIEWING_INFO_202302.csv')
VARmodel, VARtrain, VARtest, VARprediction = VARM('CI_MOVIE_VIEWING_INFO_202302.csv')
VARtrain,VARtest,VARprediction = VARtrain['MOVIE_ADNC_CO'],VARtest['MOVIE_ADNC_CO'],VARprediction['MOVIE_ADNC_CO']

model, train, test, prediction = None,None,None,None

# Button Event Function
def Command_Radio():
    global model, train, test, prediction
    if modelName.get() == 0:
        model, train, test, prediction = ARIMAmodel, ARIMAtrain, ARIMAtest, ARIMAprediction
        print("ARIMA")
    elif modelName.get() == 1:
        model, train, test, prediction = VARmodel, VARtrain, VARtest, VARprediction
        print("VAR")

def Command_Button():
    print("Clicked")
    global model, train, test, prediction
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
#label_chooseRegion = Label(window,text="choose region",font=titleFont)
#combobox_region = ttk.Combobox(window,width=15,height=10,values=region,state="readonly")
#combobox_region.set("region")
label_trainDate = Label(window,text="Trainingset Date: 2019~2022",font=titleFont)

#label_chooseRegion.place(x=10,y=10)
#combobox_region.place(x=140,y=17)
label_trainDate.place(x=10,y=10)

# Second Line, show matplotlib graph by Canvas
button_graph = Button(window,width=10,height=3,text="Show Graph",font=semi_titleFont,command=Command_Button)

button_graph.place(x=300,y=10)
## matplotlib
#figure = plt.Figure(figsize=(9,4))
#ax2 = figure.add_subplot(111)

## tkinter
#frame_graph = Frame(window,relief="flat",background="black")


# if modelName == 0 or modelName == 1:
#     line = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(figure,frame_graph)
#     line.get_tk_widget().pack(side="left")
#     train.plot(kind='line',legend=True,ax=ax2,label='Train Data')
#     test.plot(kind="line",legend=True,ax=ax2,label='Actual Data',color='orange')
#     prediction.plot(kind="line",legend=True,ax=ax2,label='Prediction',color='green')


#canvas_graph.pack(fill="both")
#frame_graph.place(x=10,y=50)

# Third Line, Input, date, quarter, model(ARIMA, OTHER MODEL), calculate button
labelframe_input = LabelFrame(window,text="Input",font=semi_titleFont)
labelframe_model = LabelFrame(labelframe_input,text="Model",font=contentFont)
labelframe_date = LabelFrame(labelframe_input,text="Date",font=contentFont)
combobox_year = ttk.Combobox(labelframe_date,width=10,height=10,values=years,state="readonly")
combobox_year.set("year")
combobox_month = ttk.Combobox(labelframe_date,width=5,height=10,values=months,state="readonly")
combobox_month.set("month")
radio_ARIMA = Radiobutton(labelframe_model,text="ARIMA",value=0,font=contentFont,variable=modelName,command=Command_Radio)
radio_other = Radiobutton(labelframe_model,text="VAR",value=1,font=contentFont,variable=modelName,command=Command_Radio)
button_calculate = Button(labelframe_input,width=10,height=3,text="Calculate",font=semi_titleFont,command=Command_Button)

combobox_year.pack(side="left",padx=5,pady=5)
combobox_month.pack(side="right",padx=5,pady=5)
labelframe_date.pack(side="left",padx=10,pady=5)
radio_ARIMA.pack(side="left",padx=5,pady=3)
radio_other.pack(side="right",padx=5,pady=3)
labelframe_model.pack(side="left",anchor='center',padx=10,pady=5)
button_calculate.pack(side="right",anchor='e',padx=10,pady=5)
labelframe_input.place(x=10,y=100)

# Fourth Line, Output, estimated customer, accuracy, '%' label
labelframe_output = LabelFrame(window,text="Output",font=semi_titleFont)
labelframe_accuracy = LabelFrame(labelframe_output,text="Accuracy",font=contentFont)
labelframe_estimatedCustomers = LabelFrame(labelframe_output,text="Estimated number of customers",font=contentFont)
entry_output = Entry(labelframe_estimatedCustomers,state='readonly',textvariable="0000000000",width=30)
entry_accuracy = Entry(labelframe_accuracy,state='readonly',textvariable="0000000000",width=15)
label_percent = Label(labelframe_accuracy,text="%")

entry_output.pack(padx=5,pady=5)
labelframe_estimatedCustomers.pack(side="left",padx=10,pady=5)
entry_accuracy.pack(side="left")
label_percent.pack(side="right")
labelframe_accuracy.pack(side='right',padx=10,pady=5)
labelframe_output.place(x=10,y=200)

window.mainloop()