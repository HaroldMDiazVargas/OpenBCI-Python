# coding=utf-8

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import lfilter,filtfilt, parzen
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from pyqtgraph.ptime import time
from PyQt5 import QtGui, QtCore
import sys; sys.path.append('..') 
import matplotlib.pyplot as plt
import pyqtgraph as pg
import HeadPlot as hp
import numpy as np
import matplotlib 
import os


def topoplotinit():
    head = hp.HeadPlot()
    return head

def topoplotfacts(head):
    head.Head(10,10,15,15)

app = QtGui.QApplication([])
pg.setConfigOption('background', 'w')

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig =  Figure(figsize=(8.0,7.0))
        self.fig.patch.set_facecolor('w')
        
        gs = GridSpec(1, 2, width_ratios=[2, 0.5])
        
        self.axes_head = self.fig.add_subplot(gs[1], xticks=([]), yticks=([]), visible=False)
        self.axes_connect = self.fig.add_subplot(gs[0], xticks=([]), yticks=([]))
        
        self.axes_head.set_visible(False)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self) # inicializa ventana ppal
        self.screenShape = QtGui.QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, self.screenShape.width(), self.screenShape.height())
        self.setStyleSheet("QMainWindow {background: 'white';}")
        self.setWindowTitle("OpenBCI_GUI")
        app_icon = QtGui.QIcon()
        app_icon.addFile('cog_1024x1024.png', QtCore.QSize(1024,1024))
        self.setWindowIcon(app_icon)
#         self.setWindowIcon(QtGui.QIcon('cog_1024x1024.png'))

        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap(QtGui.QPixmap("icon.png"))
        self.pic.setGeometry(0, 0, self.screenShape.width(), 512)  
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.show() 
    
        self.name = QtGui.QLabel("GUI - REPLAY",self)
        newfont = QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold) 
        self.name.setFont(newfont)
        self.name.resize(self.screenShape.width(),120)
        self.name.move(0,500)
        self.name.setAlignment(QtCore.Qt.AlignCenter)

        self.btn = QtGui.QPushButton("START SYSTEM", self)
        self.btn.resize(120,120)
        self.btn.move((self.screenShape.width()/2)-60,600)       
        self.btn.clicked.connect(self.home)
        self.show()
        
#--------------------------BANDPASS 1-50-------------------------------------------------------------
        self.b = [0.0579,         0,   -0.1737,         0,    0.1737,         0,   -0.0579]
        self.a = [1.0000,   -3.7335,    5.9137,   -5.2755,    2.8827,   -0.9042,    0.1180]
        
#------------------------- Notch 60 Hz----------------------------------------------------------------
        self.b_n = [0.965080986344733, -0.242468320175764, 1.94539149412878, -0.242468320175764, 0.965080986344733]
        self.a_n = [1, -0.246778261129785, 1.94417178469135, -0.238158379221743, 0.931381682126902]
        
    
    def pre_process_plot(self,eeg):
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
        main_widget.setGeometry(0, 20, self.screenShape.width()/2, self.screenShape.height()-20)
        
        headplot_widget = QtGui.QWidget(self)
        headplot_widget.setGeometry(self.screenShape.width()/2, 20, self.screenShape.width()/2, self.screenShape.height()-20)
        
        self.canvas = MplCanvas(headplot_widget)
        
        for loc, spine in self.canvas.axes_head.spines.items():
            spine.set_linewidth(0)
            
        for loc, spine in self.canvas.axes_connect.spines.items():
            spine.set_linewidth(0)
            
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('EEG Data System')
        pg.setConfigOptions(antialias=True)

#--------------------PARAMETERS----------------------------------------------        
        
        self.time = 0.4
        self.fs =  250
        self.channels = 8
        self.win_size = 4

#-------------------INIT HEAD---------------------------------------------
        
        self.head = topoplotinit()
        topoplotfacts(self.head)
        
        self.xy_center = [self.head.circ_x,self.head.circ_y]
        self.radius = self.head.circ_diam/2 
        xy_center_earR = [self.head.earR_x,self.head.earR_y]
        xy_center_earL = [self.head.earL_x,self.head.earL_y]
        
        xy_nose = [[self.head.nose_x[0],self.head.nose_y[0]],[self.head.nose_x[1],self.head.nose_y[1]],[self.head.nose_x[2],self.head.nose_y[2]]]

        
#-------------------HEAD---------------------------------------------  
  
        circle_earR = matplotlib.patches.Ellipse(xy = xy_center_earR, width = self.head.ear_width, height = self.head.ear_height, angle = 0, edgecolor = "k", facecolor = "none", zorder = 0) 
        self.canvas.axes_connect.add_patch(circle_earR) 
        circle_earL = matplotlib.patches.Ellipse(xy = xy_center_earL, width = self.head.ear_width, height = self.head.ear_height, angle = 0, edgecolor = "k", facecolor = "none", zorder = 0) 
        self.canvas.axes_connect.add_patch(circle_earL) 
        circle = matplotlib.patches.Circle(xy = self.xy_center, radius = self.radius, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(circle) 
        polygon = matplotlib.patches.Polygon(xy = xy_nose, edgecolor = "k", facecolor = "w",zorder = 0) 
        self.canvas.axes_connect.add_patch(polygon)  
        elec_1 = matplotlib.patches.Circle(xy = self.head.electrode_xy[0], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_1)
        self.canvas.axes_connect.annotate("Fp1", xy = self.head.electrode_xy[0], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_2 = matplotlib.patches.Circle(xy = self.head.electrode_xy[1], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_2)
        self.canvas.axes_connect.annotate("Fp2", xy = self.head.electrode_xy[1], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_3 = matplotlib.patches.Circle(xy = self.head.electrode_xy[2], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_3)
        self.canvas.axes_connect.annotate("F3", xy = self.head.electrode_xy[2], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_4 = matplotlib.patches.Circle(xy = self.head.electrode_xy[3], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_4) 
        self.canvas.axes_connect.annotate("F4", xy = self.head.electrode_xy[3], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_5 = matplotlib.patches.Circle(xy = self.head.electrode_xy[4], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_5) 
        self.canvas.axes_connect.annotate("P7", xy = self.head.electrode_xy[4], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_6 = matplotlib.patches.Circle(xy = self.head.electrode_xy[5], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_6) 
        self.canvas.axes_connect.annotate("P8", xy = self.head.electrode_xy[5], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_7 = matplotlib.patches.Circle(xy = self.head.electrode_xy[6], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_7) 
        self.canvas.axes_connect.annotate("O1", xy = self.head.electrode_xy[6], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")
        elec_8 = matplotlib.patches.Circle(xy = self.head.electrode_xy[7], radius = self.head.elec_diam, edgecolor = "k", facecolor = "w") 
        self.canvas.axes_connect.add_patch(elec_8) 
        self.canvas.axes_connect.annotate("O2", xy = self.head.electrode_xy[7], fontsize=2*self.head.elec_diam, ha="center", color = "dimgrey")


        self.cb=True

        self.canvas.axes_connect.set_xlim(self.head.image_x - 1, self.radius + self.head.circ_x + self.head.ear_width/2 + 1) 
        self.canvas.axes_connect.set_ylim(self.radius + self.head.circ_y + 1,self.head.image_y)
        
#---------------GRAPHS-----------------------------------------------------

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
        
        self.btnStart = QtGui.QPushButton("Start", self)
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
        
        self.btnFiles = QtGui.QPushButton("Select Files",self)
        self.btnFiles.move(280,0)
        self.btnFiles.resize(130,25)
        self.btnFiles.clicked.connect(self.Files)
                
#------------------------------LABELS----------------------------------------
        
        self.LblScaleVert = QtGui.QLabel("VertScale", self)
        self.LblScaleVert.move(200,20)
        self.lblFiles= QtGui.QLabel("Select Files", self)
        self.lblFiles.move(280,20)
        self.lblFiles.resize(130,30)
        self.lblFilter = QtGui.QLabel("Filter", self)
        self.lblFilter.move(120,20)
                
#-------------------FIRST RECORD---------------------------------------------
        a = 1
        
        for path in QtGui.QFileDialog.getOpenFileNames(self, "Select File To Load"): 
            if a < 2:
                list_path = str(path)
            a = a+1
        a = 1   
        length = len(list_path)
        new_path = list_path[2:length-2]
        self.eeg = np.loadtxt(new_path,delimiter=' ',skiprows=2)
        y1 = self.eeg[:8,:100]

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

        self.counter = self.time
        self.idx = 0
        
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
            self.lblFiles.hide()
            self.btnFiles.hide()
            self.btnStart.setText("Stop")
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update)
            self.timer.start(400)
        else:
            self.lblFiles.show()
            self.btnFiles.show()
            self.btnStart.setText("Start")
            self.timer.stop()   
    
    def Files(self):
        a = 1
        for path in QtGui.QFileDialog.getOpenFileNames(self, "Select File To Load"):
            if a < 2:
                list_path = str(path)
            a = a+1
        a = 1   
        length = len(list_path)
        new_path = list_path[2:length-2]
        self.eeg = np.loadtxt(new_path,delimiter=' ',skiprows=2)  
        self.y = self.eeg[:8,:100]
        self.idx=0
        self.counter = self.time 

    def update(self):
        global y_p


        if (self.idx < self.eeg.shape[1]/100):

            y2 = self.eeg[:8,(self.idx*100):((self.idx+1)*100)]

            #--------------------------------------------------------
            #EEG Plot
            if np.ceil(self.counter) < self.win_size: 
                self.y = np.c_[self.y,y2] 
                self.x = self.xT[:self.y.shape[1]] 
                y_f = self.pre_process_plot(self.y) 
                y_p = y_f 
            else:
                self.y = np.c_[self.y,y2]  
                self.y = self.y[:,int(self.time*self.fs):]  
                self.x = self.xT[:self.y.shape[1]]
                y_f = self.pre_process_plot(self.y) 
                y_p = np.c_[y_p,y_f[:,(y_f.shape[1]-int(self.time*self.fs)):y_f.shape[1]]]
                y_p = y_p[:,int(self.time*self.fs):]

            for i in range(self.channels):
                self.curve[i].setData(self.x,y_p[i], pen='k') 
                self.canvas.draw()

            self.counter = self.counter + self.time 
            self.idx += 1

               
if __name__ == '__main__':
	import sys
	app_window = MainWindow()
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()

