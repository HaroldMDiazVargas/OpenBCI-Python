# coding=utf-8



from scipy.signal import butter, filtfilt, parzen, lfilter
from datetime import datetime, date
from pyqtgraph.ptime import time
from PyQt5 import QtGui, QtCore
import sys; sys.path.append('..') 
import open_bci as bci 
import pyqtgraph as pg
import numpy as np 
import time as tm
import os


def saveData(sample):
    """Add the channels value to eeg variable
    """
    global eeg
    eeg.append(sample.channel_data)


    
def connect_board():
    """Connect with openbci board
    Create a object from class "OpenBCIBoard" to handle the data sending
    """
    baud = 115200
    board = bci.OpenBCIBoard(port=None, baud = baud, filter_data = True)
    print("Board Connected")
    return board



def initialize(board):
    """Initialize the board
    Active the stream mode
    """
    global eeg
    eeg = []
    tm.sleep(1)
    board.ser.write(b'v')
    tm.sleep(1)
    #board.enable_filters()
#     tm.sleep(0.1)
    board.start_streaming(saveData)
    print('Board initializated')

    
def disconnect_board(board):
    """Disconnect the board
    Record the last sample but doesn't save it
    """
    global eeg
    eeg = []
    board.ser.write('v')
    tm.sleep(0.1)
    board.start_streaming(saveData)
    print('Streaming ended')
    print('')
    board.disconnect()
    sys.exit()

app = QtGui.QApplication([])
pg.setConfigOption('background', 'w')

class MainWindow(QtGui.QMainWindow):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    def __init__(self):
#         
        QtGui.QMainWindow.__init__(self)
    
        self.screenShape = QtGui.QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, self.screenShape.width(), self.screenShape.height())
        self.setStyleSheet("QMainWindow {background: 'white';}")
        self.setWindowTitle("OpenBCI_GUI")
        app_icon = QtGui.QIcon()
        app_icon.addFile('cog_1024x1024.png', QtCore.QSize(1024,1024))
        self.setWindowIcon(app_icon)


        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap(QtGui.QPixmap("icon.png"))
        self.pic.setGeometry(0, 0, self.screenShape.width(), 512)  
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.show() 
    
        self.name = QtGui.QLabel("GUI RECORD",self)
        newfont = QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold) 
        self.name.setFont(newfont)
        self.name.resize(self.screenShape.width(),120)
        self.name.move(0,500)
        self.name.setAlignment(QtCore.Qt.AlignCenter)

        self.btn = QtGui.QPushButton("StartSystem", self)
        self.btn.resize(120,120)
        self.btn.move((self.screenShape.width()/2)-60,600)       
        self.btn.clicked.connect(self.home)
        self.show()

        
#--------------------------BANDPASS 1-50-------------------------------------------------------------

        self.b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]
        self.a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
        
#------------------------- Notch 60 Hz---------------------------------------------------------------
        self.b_n = [0.965080986344733, -0.242468320175764, 1.94539149412878, -0.242468320175764, 0.965080986344733]
        self.a_n = [1, -0.246778261129785, 1.94417178469135, -0.238158379221743, 0.931381682126902]
        
        self.keyPressed.connect(self.on_key)
    
    def keyPressEvent(self, event):
        super(MainWindow, self).keyPressEvent(event)
        self.keyPressed.emit(event)
        
    def get_n_secs(self,board,n):
        global eeg
        eeg = []

        for i in range(int(round(n*250))):
            board.start_streaming(saveData)
            self.counter+=self.Ts

        return(eeg)
    
    def pre_process_plot(self,eeg):
        """Scale and center the data
        Then, apply them a notch and band-pass filter
        """
        eeg = np.array(eeg)
        [fil,col] = eeg.shape
        eeg_processed = np.zeros([fil,col])
        for i in range(fil):
            data = eeg[i,:] * 2.23517444553071e-02
            data = data - np.mean(data)
            data = lfilter(self.b_n,self.a_n, data)
            data = lfilter(self.b,self.a,data)       
            eeg_processed[i,:] = data
        return(eeg_processed)
    
    def home(self):
       
        self.pic.hide() 
        self.name.hide()
        self.btn.hide()
        self.hide() 
        
        main_widget = QtGui.QWidget(self)
        main_widget.setGeometry(0, 20, self.screenShape.width(), self.screenShape.height()-20)
                
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('EEG Data System')
        pg.setConfigOptions(antialias=True)

#--------------------PARAMETERS----------------------------------------------        
        
        self.stim=[] #Save the stim
        self.time = 0.4
        self.Ts = 4 
        self.fs =  250
        self.counter=-4.0
        self.channels = 8
        self.win_size = 4
        self.datetime = ""
        
#---------------INTERFACE-----------------------------------------------------

        #EEG Plot
        self.plt8 = win.addPlot(title="O2",row=7, col=0, colspan=2)
        self.plt8.setWindowTitle('O2')
        self.plt8.setYRange(-50, 50)
        self.plt8.setXRange(0, self.win_size-1)

        self.plt7 = win.addPlot(title="O1",row=6, col=0, colspan=2)
        self.plt7.setWindowTitle('O1')
        self.plt7.setYRange(-50, 50)
        self.plt7.setXRange(0, self.win_size-1)
        self.plt7.hideAxis('bottom')

        self.plt6 = win.addPlot(title="P8",row=5, col=0, colspan=2)
        self.plt6.setWindowTitle('P8')
        self.plt6.setYRange(-50, 50)
        self.plt6.setXRange(0, self.win_size-1)
        self.plt6.hideAxis('bottom')
        
        self.plt5 = win.addPlot(title="P7",row=4, col=0, colspan=2)
        self.plt5.setWindowTitle('P7')
        self.plt5.setYRange(-50, 50)
        self.plt5.setXRange(0, self.win_size-1)
        self.plt5.hideAxis('bottom')

        self.plt4 = win.addPlot(title="F4",row=3, col=0, colspan=2)
        self.plt4.setWindowTitle('C4')
        self.plt4.setYRange(-50, 50)
        self.plt4.setXRange(0, self.win_size-1)
        self.plt4.hideAxis('bottom')

        self.plt3 = win.addPlot(title="F3",row=2, col=0, colspan=2)
        self.plt3.setWindowTitle('C3')
        self.plt3.setYRange(-50, 50)
        self.plt3.setXRange(0, self.win_size-1)
        self.plt3.hideAxis('bottom')

        self.plt2 = win.addPlot(title="Fp2",row=1, col=0, colspan=2)
        self.plt2.setWindowTitle('FP2')
        self.plt2.setYRange(-50,50)
        self.plt2.setXRange(0, self.win_size-1)
        self.plt2.hideAxis('bottom')

        self.plt1 = win.addPlot(title="Fp1",row=0, col=0, colspan=2)
        self.plt1.setWindowTitle('FP1')
        self.plt1.setLabel('bottom', 'Time', units='sec')
        self.plt1.setYRange(-50, 50)
        self.plt1.setXRange(0, self.win_size-1)
        self.plt1.hideAxis('bottom')

#----------------------BUTTONS------------------------------------------------
        
        self.btnStart = QtGui.QPushButton("StartDataStream", self)
        self.btnStart.move(0,0)
        self.btnStart.resize(120,25)
        self.btnStart.clicked.connect(self.Start)
        self.stream = False
        
        self.btnFilter = QtGui.QPushButton("BP:5-45",self)
        self.btnFilter.move(120,0)
        self.btnFilter.resize(self.btnFilter.minimumSizeHint())
        self.btnFilter.clicked.connect(self.Filter)
        self.filt=1
        
        btnScaleVert = QtGui.QComboBox(self)
        btnScaleVert.move(200,0)
        btnScaleVert.resize(self.btnFilter.minimumSizeHint())
        btnScaleVert.addItem("50uV")
        btnScaleVert.addItem("100uV")
        btnScaleVert.addItem("200uV")
        btnScaleVert.addItem("400uV") 
        btnScaleVert.activated[str].connect(self.ScaleVert)
                
#------------------------------LABELS----------------------------------------
        
        self.LblScaleVert = QtGui.QLabel("VertScale", self)
        self.LblScaleVert.move(200,20)
        self.lblFilter = QtGui.QLabel("Filter", self)
        self.lblFilter.move(120,20)
                
#-------------------FIRST RECORD---------------------------------------------

        self.board = connect_board()
        initialize(self.board)
        
        eeg = self.get_n_secs(self.board,self.time)
        eeg = np.asarray(eeg)
        y1 = np.transpose(eeg.tolist())
        
        y_p1 = self.pre_process_plot(y1)

        x1 = np.linspace(0,self.time,int(self.fs*self.time))
        
        self.curve = []
        c = self.plt1.plot(x1, y_p1[0], pen='w')
        self.curve.append(c)
        c = self.plt2.plot(x1, y_p1[1], pen='w')  
        self.curve.append(c)
        c = self.plt3.plot(x1, y_p1[2], pen='w')  
        self.curve.append(c)
        c = self.plt4.plot(x1, y_p1[3], pen='w')  
        self.curve.append(c)
        c = self.plt5.plot(x1, y_p1[4], pen='w') 
        self.curve.append(c)
        c = self.plt6.plot(x1, y_p1[5], pen='w')  
        self.curve.append(c)
        c = self.plt7.plot(x1, y_p1[6], pen='w') 
        self.curve.append(c)
        c = self.plt8.plot(x1, y_p1[7], pen='w') 
        self.curve.append(c)
        
        self.x = np.linspace(0,self.win_size,self.fs*self.win_size)
        self.xT = np.linspace(0,self.win_size,self.fs*self.win_size)
        self.y = y1
        
        box_layout = QtGui.QVBoxLayout(main_widget) 
        box_layout.addWidget(win)
        self.show()        
        
        
    def ScaleVert(self, text):
        
        if (text=="50uV"):
            self.plt1.setYRange(-50, 50)
            self.plt2.setYRange(-50, 50)
            self.plt3.setYRange(-50, 50)
            self.plt4.setYRange(-50, 50)
            self.plt5.setYRange(-50, 50)
            self.plt6.setYRange(-50, 50)
            self.plt7.setYRange(-50, 50)
            self.plt8.setYRange(-50, 50)
        
        elif (text=="100uV"):
            self.plt1.setYRange(-100, 100)
            self.plt2.setYRange(-100, 100)
            self.plt3.setYRange(-100, 100)
            self.plt4.setYRange(-100, 100)
            self.plt5.setYRange(-100, 100)
            self.plt6.setYRange(-100, 100)
            self.plt7.setYRange(-100, 100)
            self.plt8.setYRange(-100, 100)
            
        elif (text=="200uV"):
            self.plt1.setYRange(-200, 200)
            self.plt2.setYRange(-200, 200)
            self.plt3.setYRange(-200, 200)
            self.plt4.setYRange(-200, 200)
            self.plt5.setYRange(-200, 200)
            self.plt6.setYRange(-200, 200)
            self.plt7.setYRange(-200, 200)
            self.plt8.setYRange(-200, 200)
        else: 
            self.plt1.setYRange(-400, 400)
            self.plt2.setYRange(-400, 400)
            self.plt3.setYRange(-400, 400)
            self.plt4.setYRange(-400, 400)
            self.plt5.setYRange(-400, 400)
            self.plt6.setYRange(-400, 400)
            self.plt7.setYRange(-400, 400)
            self.plt8.setYRange(-400, 400)
            
        
    def Filter(self):
        
        self.filt+=1
        
        if(self.filt>4): 
            self.filt=0
        else: 
            pass
        
        if(self.filt==0):
            self.b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]
            self.a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
            self.btnFilter.setText("BP:5-45")     
        
        elif(self.filt==1):

            self.b = [ 0.200138725658073, 0, -0.400277451316145, 0, 0.200138725658073 ]
            self.a = [ 1, -2.35593463113158, 1.94125708865521, -0.784706375533419, 0.199907605296834 ]
            self.btnFilter.setText("BP:1-50")         
            
        elif(self.filt==2):
            self.b = [ 0.00512926836610803, 0, -0.0102585367322161, 0, 0.00512926836610803 ]
            self.a = [ 1, -3.67889546976404, 5.17970041352212, -3.30580189001670, 0.807949591420914 ]
            self.btnFilter.setText("BP:7-13")
            
        elif(self.filt==3):
            self.b = [ 0.117351036724609, 0, -0.234702073449219, 0, 0.117351036724609  ]
            self.a = [ 1, -2.13743018017206, 2.03857800810852, -1.07014439920093, 0.294636527587914 ] 
            self.btnFilter.setText("BP:15-50")
            print(self.filt)
            
        else:
            self.b = [ 0.175087643672101, 0, -0.350175287344202, 0, 0.175087643672101  ]
            self.a = [ 1, -2.29905535603850, 1.96749775998445, -0.874805556449481, 0.219653983913695 ] 
            self.btnFilter.setText("BP:5-50")
           
    def Start(self):
        
        self.stream = not self.stream

        if (self.stream):
                self.btnStart.setText("StopDataStream")
                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self.update)
                self.timer.start(0)
                self.datetime=datetime.now().strftime('Date:%Y-%m-%d_Time:%H:%M:%S')
        else: 
            self.btnStart.setText("StartDataStream")
            self.timer.stop()
            
            filename=input('Save as (without extension): \n(write exit to exit without saving) ')
            
            if (filename!="exit"):
                aux=np.zeros((2,self.raw_data.shape[1]))
                self.raw_data=np.r_[self.raw_data,aux] 

                print("Storing data...")

                for i in range (self.raw_data.shape[1]):# Time vector
                    self.raw_data[-2][i]=(i*4.0)/1000.0

                for i in range(len(self.stim)):
                    self.raw_data[-1][int((self.stim[i]/4)*1000)]=1 # Stim vector
            
                np.savetxt('SavedData/'+filename+'.csv',self.raw_data,header=self.datetime+"\n"+"Channels:Fp1,FP2,F3,F4,P7,P8,O1,O2",comments='')
                print("Stored data in: "+os.getcwd()+"/SavedData/"+filename+'.csv')
                print(self.stim)
            
            else:
                print("No stored data")

            eeg = self.get_n_secs(self.board,self.time)
            eeg = np.asarray(eeg)
            self.y = np.transpose(eeg.tolist())
            self.counter=-4.0
            self.stim=[]
       
    
    def update(self):
        global y_p

        EEG_new = self.get_n_secs(self.board,self.time)
        EEG_new = np.asarray(EEG_new)
        y2 = np.transpose(EEG_new.tolist())

        #--------------------------------------------------------
        #EEG Plot
        if np.ceil(self.counter/1000) < self.win_size:
            self.y = np.c_[self.y,y2]
            self.raw_data = self.y
            self.x = self.xT[:self.y.shape[1]]
            y_f = self.pre_process_plot(self.y)
            y_p = y_f
        else:
            self.raw_data = np.c_[self.raw_data,y2]
            self.y = np.c_[self.y,y2]
            self.y = self.y[:,int(self.time*self.fs):]
            self.x = self.xT[:self.y.shape[1]]
            y_f = self.pre_process_plot(self.y)
            y_p = np.c_[y_p,y_f[:,(y_f.shape[1]-int(self.time*self.fs)):y_f.shape[1]]]
            y_p = y_p[:,int(self.time*self.fs):]

        for i in range(self.channels):
            self.curve[i].setData(self.x,y_p[i], pen='k') 
            
    def on_key(self, event):
        if event.key() == QtCore.Qt.Key_S:
            self.stim.append(self.counter/1000)

               
if __name__ == '__main__':
    import sys
    app_window = MainWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    disconnect_board(app_window.board)

