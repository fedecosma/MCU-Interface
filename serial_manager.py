import serial
import serial.tools.list_ports
from config import BAUDRATE
from utils import crc8_sum

class SerialManager:
    def __init__(self, app):
        self.app = app
        self.serial = None

    #refresh list of available ports
    def refresh_ports(self):
        self.app.port_cb.clear()
        for p in serial.tools.list_ports.comports():
            self.app.port_cb.addItem(p.device)

    #connect to a COM port
    def connect_serial(self, port):
        try:
            self.serial = serial.Serial(port, baudrate=BAUDRATE, timeout=0.1)
            self.app.status_label.setText(f"Connected")
            self.app.log.append(f"Connected to {port}")
            self.app.timer.start(30)
            self.send_cmd([0x35, 0x01, 0x00])   #get status on connection in order to immediately update settings' global vars
            self.app.thread_running = True  #starts thread as soon as connected, for sampling temperature and humidity every 4 seconds
            self.app.thread_th.start()
            return True
        except Exception as e:
            self.app.log.append(f"Error opening serial port: {e}")
            return False

    #disconnect from COM port
    def disconnect_serial(self):
        if self.serial and self.serial.is_open:
            self.app.thread_running = False
            if self.app.thread_th.is_alive():   #stops thread
                self.app.thread_th.join()
            self.serial.close()
            self.app.timer.stop()
            self.app.status_label.setText("Disconnected")
            self.app.log.append("Disconnected.")

    #sends command
    def send_cmd(self, bytes_wo_crc: list[int]):
        if not self.serial or not self.serial.is_open:
            self.app.log.append("Serial not connected.")
            return
        if self.app.streaming and bytes_wo_crc[0] not in (0x37, 0x38, 0x42): #while streaming/acquiring only enable start stream, stop stream and temperature/humidity request (0x42)
            return
        if bytes_wo_crc[0] == 0x37:
            self.app.streaming = True
        if bytes_wo_crc[0] == 0x38:
            self.app.streaming = False

        if bytes_wo_crc[0] == 0x36:                     #when setup cmd is sent, update global variables of current setup
            self.app.current_odr = bytes_wo_crc[2]
            self.app.current_fs_code = bytes_wo_crc[3]
            self.app.current_axes =  bytes_wo_crc[4]

        pkt = bytes_wo_crc + [crc8_sum(bytes_wo_crc)]
        self.serial.write(bytearray(pkt))
        if not self.app.thread_req:                                             #if it's not a thread request (0x42) prints what has been transmitted
            self.app.log.append(f"TX: {' '.join(f'0x{x:02X}' for x in pkt)}")
