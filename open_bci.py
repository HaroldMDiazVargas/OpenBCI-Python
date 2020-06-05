# coding=utf-8
"""
Core OpenBCI object for handling connections and samples from the board.

EXAMPLE USE:

def handle_sample(sample):
  print(sample.channel_data)

board = OpenBCIBoard()
board.print_register_settings()
board.start_streaming(handle_sample)

NOTE: If daisy modules is enabled, the callback will occur every two samples, hence "packet_id" will only contain even numbers. As a side effect, the sampling rate will be divided by 2.

FIXME: at the moment we can just force daisy mode, do not check that the module is detected.
TODO: enable impedance

"""
import numpy as np
import logging
import serial
import struct
import atexit
import glob
import time
import sys


SAMPLE_RATE = 250.0 
START_BYTE = 0xA0  
END_BYTE = 0xC0  

'''
#Commands for in SDK http://docs.openbci.com/software/01-Open BCI_SDK:

command_stop = "s";
command_startText = "x";
command_startBinary = "b";
command_startBinary_wAux = "n";
command_startBinary_4chan = "v";
command_activateFilters = "F";
command_deactivateFilters = "g";
command_deactivate_channel = {"1", "2", "3", "4", "5", "6", "7", "8"};
command_activate_channel = {"q", "w", "e", "r", "t", "y", "u", "i"};
command_activate_leadoffP_channel = {"!", "@", "#", "$", "%", "^", "&", "*"};  //shift + 1-8
command_deactivate_leadoffP_channel = {"Q", "W", "E", "R", "T", "Y", "U", "I"};   //letters (plus shift) right below 1-8
command_activate_leadoffN_channel = {"A", "S", "D", "F", "G", "H", "J", "K"}; //letters (plus shift) below the letters below 1-8
command_deactivate_leadoffN_channel = {"Z", "X", "C", "V", "B", "N", "M", "<"};   //letters (plus shift) below the letters below the letters below 1-8
command_biasAuto = "`";
command_biasFixed = "~";
'''

class OpenBCIBoard(object):
    def __init__(self, port=None, baud=115200, filter_data=True,timeout=None):
        
        """
        Making the connection with the OpenBCI board.
        Defines functions to unpack data, comunnicate
        with the board, start and stop the data stream.

        """
        self.streaming = False
        self.baudrate = baud
        self.timeout = timeout

        if not port:
            port = self.find_port()

        self.port = port

        print("Connecting to V3 at port %s" %(port))
        self.ser = serial.Serial(port= port, baudrate = baud, timeout=timeout)

#         time.sleep(2)
#         self.ser.write(b'v');
#         time.sleep(1)

        self.filtering_data = filter_data
        self.eeg_channels_per_sample = 8 
        self.aux_channels_per_sample = 3 
        self.read_state = 0
        self.attempt_reconnect = False
        self.last_reconnect = 0
        self.reconnect_freq = 5
        self.packets_dropped = 0
        atexit.register(self.disconnect)
        
        
        
        
    def start_streaming(self, callback):

        if not self.streaming:

            self.ser.write(b'b')
            self.streaming = True

        if not isinstance(callback, list):
            callback = [callback]

        sample = self._read_serial_binary()


        for call in callback:
            call(sample) 


    def _read_serial_binary(self, max_bytes_to_skip=10000):

        def read(n):
            b = self.ser.read(n)
            if not b:

                self.warn('Device appears to be stalled. Quitting...')
                sys.exit()
                raise Exception('Device Stalled')
                sys.exit()
                return '\xFF'
            else:
                return b


        for rep in range(max_bytes_to_skip):
            #---------Start Byte & ID---------
            if self.read_state == 0:
                b = read(1)
                if struct.unpack('B', b)[0] == START_BYTE:
                    if(rep != 0):
                        #self.warn('Skipped %d bytes before start found' %(rep))
                        rep = 0;
                    packet_id = struct.unpack('B', read(1))[0] 
                    self.read_state = 1

            #---------Channel Data---------
            elif self.read_state == 1:
                channel_data = []
                for c in range(self.eeg_channels_per_sample):
                    literal_read = read(3)
                    unpacked = struct.unpack('3B', literal_read)

                    if (unpacked[0] > 127):
                        pre_fix = bytes(bytearray.fromhex('FF')) 
                    else:
                        pre_fix = bytes(bytearray.fromhex('00'))


                    literal_read = pre_fix + literal_read;
                    myInt = struct.unpack('>i', literal_read)[0]
                    channel_data.append(myInt)

                self.read_state = 2;

            #---------Accelerometer Data---------
            elif self.read_state == 2:
                aux_data = []
                for a in range(self.aux_channels_per_sample):
                    acc = struct.unpack('>h', read(2))[0]
                    aux_data.append(acc)

                self.read_state = 3;
            elif self.read_state == 3:
                val = struct.unpack('B', read(1))[0]
                self.read_state = 0 # Lea el siguiente paquete si no está el end byte.
                if (val == END_BYTE):
                    sample = OpenBCISample(packet_id, channel_data, aux_data)
                    self.packets_dropped = 0
                    return sample
                else:
                    self.packets_dropped = self.packets_dropped + 1

    def stop(self):
        self.streaming = False
        self.ser.write(b's')


    def disconnect(self):
        if(self.streaming == True):
            self.stop()
        if (self.ser.isOpen()):
            print("Closing Serial...")
            self.ser.close()
            logging.warning('serial closed')

    def warn(self, text):
        print("Warning: %s" % text)


    def find_port(self):
        print('Searching Board...')
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = serial.Serial(port= port, baudrate = self.baudrate, timeout=self.timeout)
                s.write(b'v')
                #print(s)
                openbci_serial = self.openbci_id(s)
                s.close()
                if openbci_serial:
                    openbci_port = port;
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port')
        else:
            return openbci_port


    def openbci_id(self, serial):
        line = ''
        time.sleep(2)

        if serial.inWaiting():
            line = ''
            c = ''
            while '$$$' not in line:
                c = serial.read().decode('utf-8')
                line += c
        print("Serial established...")
        print(line)
        if "OpenBCI" in line:
            return True
        return False

class OpenBCISample(object):
    def __init__(self, packet_id, channel_data, aux_data):
        """
        Encapsulates each of the OpenBci data samples.
         """

        self.id = packet_id;
        self.channel_data = channel_data;
        self.aux_data = aux_data;


