import threading
import os
import time
import csv
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QCheckBox, QGroupBox, QSizePolicy
)
from PyQt6.QtCore import QTimer
from config import ODR_TEXT, FS_TEXT
from serial_manager import SerialManager
from parser import FrameParser
from utils import axes_text

class SerialApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCU Interface")
        self.resize(2000, 700)

        # Serial manager and parser
        self.serial_manager = SerialManager(self)
        self.parser = FrameParser(self)

        # Status
        self.streaming = False
        self.acquisition = False
        self.rx_buffer = bytearray()
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)

        # Thread polling temperature/humidity
        self.thread_req = False
        self.thread_running = False
        self.thread_th = threading.Thread(target=self.poll_temp_hum, daemon=True)

        # Global vars for current settings and temp/hum.
        self.current_fs_code = 0x30     #default settings
        self.current_axes = 0x07
        self.current_odr = 0x50
        self.current_temperature = 0.0
        self.current_humidity = 0.0

        # Csv
        self.csv_file = None
        self.csv_writer = None
        self.acquisition_timer = None  #for actual acquisition
        self.acquiring_timer = None #for acquiring animation in log
        self.acquiring_dots = 0

        # UI
        self.build_ui()
        self.serial_manager.refresh_ports()

    # ---------------- GUI ----------------
    def build_ui(self):
        main_layout = QHBoxLayout()

        #---left side---
        left_col = QVBoxLayout()

        #serial port
        port_layout = QHBoxLayout()
        self.port_cb = QComboBox()

        #refresh
        btn_refresh = QPushButton("Refresh Ports")
        btn_refresh.clicked.connect(self.serial_manager.refresh_ports)

        #connect
        btn_connect = QPushButton("Connect")
        btn_connect.clicked.connect(lambda: self.serial_manager.connect_serial(self.port_cb.currentText()))

        #disconnect
        btn_disconnect = QPushButton("Disconnect")
        btn_disconnect.clicked.connect(self.serial_manager.disconnect_serial)

        port_layout.addWidget(QLabel("Serial Port:"))
        port_layout.addWidget(self.port_cb)

        port_layout.addWidget(btn_refresh)
        port_layout.addWidget(btn_connect)
        port_layout.addWidget(btn_disconnect)

        left_col.addLayout(port_layout)

        self.status_label = QLabel("Disconnected")
        left_col.addWidget(self.status_label)

        #commands group
        cmd_group = QGroupBox("Commands")
        cmd_layout = QVBoxLayout()

        # 0x35 - check accelerometer status
        btn35 = QPushButton("Check Accelerometer (Status)")
        btn35.setMaximumWidth(230)
        btn35.clicked.connect(lambda: self.serial_manager.send_cmd([0x35, 0x01, 0x00]))
        cmd_layout.addWidget(btn35)

        # accelerometer setup
        setup_box = QGroupBox("Setup Accelerometer")
        setup_layout = QHBoxLayout()

        #ODR list
        self.rate_cb = QComboBox()
        for val, txt in ODR_TEXT.items():
            self.rate_cb.addItem(txt, val)
        self.rate_cb.setCurrentText("100 Hz")

        #full scale list
        self.fs_cb = QComboBox()
        for val, txt in FS_TEXT.items():
            self.fs_cb.addItem(txt, val)
        self.fs_cb.setCurrentText("±16 g")

        #axes
        self.chk_x = QCheckBox("X"); self.chk_x.setChecked(True)
        self.chk_y = QCheckBox("Y"); self.chk_y.setChecked(True)
        self.chk_z = QCheckBox("Z"); self.chk_z.setChecked(True)

        btn36 = QPushButton("Send Setup")
        btn36.setMaximumWidth(230)
        btn36.clicked.connect(self.send_setup)
        for w in [QLabel("Data Rate:"), self.rate_cb, QLabel("Full Scale:"), self.fs_cb,
                  self.chk_x, self.chk_y, self.chk_z, btn36]:
            setup_layout.addWidget(w)

        setup_box.setLayout(setup_layout)
        cmd_layout.addWidget(setup_box)

        #acquisition
        acq_box = QGroupBox("Acquisition")
        acq_layout = QHBoxLayout()

        self.timeframe_cb = QComboBox()
        self.timeframe_cb.addItems(["1 min", "5 min", "10 min", "15 min", "30 min"])

        btn37 = QPushButton("Start Acquisition")
        btn37.setMaximumWidth(230)
        btn37.clicked.connect(self.start_acquisition)

        btn38 = QPushButton("Stop Acquisition")
        btn38.setMaximumWidth(230)
        btn38.clicked.connect(self.stop_acquisition)

        for w in [QLabel("Timeframe:"), self.timeframe_cb, btn37, btn38]:
            acq_layout.addWidget(w)

        acq_box.setLayout(acq_layout)
        cmd_layout.addWidget(acq_box)

        #other commands
        commands = [
            ("Start Stream", [0x37, 0x01, 0x00]),
            ("Stop Stream", [0x38, 0x01, 0x00]),
            ("Read Acceleration Sample", [0x39, 0x01, 0x00]),
            ("Read Acceleration (A_ms)", [0x40, 0x01, 0x00]),
            ("Read Acceleration (A_mt)", [0x41, 0x01, 0x00]),
            ("Read Temperature/Humidity", [0x42, 0x01, 0x00]),
            ("Reset Max Acceleration", [0x43, 0x01, 0x00]),
        ]
        for label, payload in commands:
            b = QPushButton(label)
            b.setMaximumWidth(230)
            b.clicked.connect(lambda _, p=payload: self.serial_manager.send_cmd(p))
            cmd_layout.addWidget(b)

        #set date and time
        btn50 = QPushButton("Set Date/Time")
        btn50.setMaximumWidth(230)
        btn50.clicked.connect(self.send_datetime_now)
        cmd_layout.addWidget(btn50)

        cmd_group.setLayout(cmd_layout)
        left_col.addWidget(cmd_group)
        left_col.addStretch()
        main_layout.addLayout(left_col, 0)

        #---right side--- (log)
        log_box = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log.setSizePolicy(policy)

        btn_clear = QPushButton("Clear Data")
        btn_clear.clicked.connect(self.log.clear)

        log_box.addWidget(self.log, 1)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        log_box.addLayout(btn_row)

        main_layout.addLayout(log_box, 1)

        self.setLayout(main_layout)

    #clicked buttons' functions
    def send_setup(self):
        rate = self.rate_cb.currentData()
        fs = self.fs_cb.currentData()
        axes = (0x01 if self.chk_x.isChecked() else 0) | \
               (0x02 if self.chk_y.isChecked() else 0) | \
               (0x04 if self.chk_z.isChecked() else 0)
        self.serial_manager.send_cmd([0x36, 0x03, rate, fs, axes])

    def start_acquisition(self):
        if not self.serial_manager.serial or not self.serial_manager.serial.is_open:
            self.log.append("Serial not connected.")
            return
        if self.streaming: #if already streaming no action
            return
        self.streaming = True
        self.acquisition = True

        minutes = int(self.timeframe_cb.currentText().split()[0]) #get selected timeframe
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"acquisition_{now}.csv"
        filepath = os.path.join("acquisitions", filename)

        self.csv_file = open(filepath, "w", newline = "")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Acquisition settings", "Timeframe", "Data Rate", "Full Scale", "Enabled Axes"])
        self.csv_writer.writerow(["", f"{minutes} min", ODR_TEXT.get(self.current_odr, f"0x{self.current_odr:02X}"), FS_TEXT.get(self.current_fs_code, f"0x{self.current_fs_code:02X}"), axes_text(self.current_axes)])
        self.csv_writer.writerow(["Date","Time","Acc_X [g]","Acc_Y [g]","Acc_Z [g]","Temperature [°C]","Humidity [%]"])

        self.serial_manager.send_cmd([0x37, 0x01, 0x00])

        self.acquiring_dots = 0
        self.acquiring_timer = QTimer()
        self.acquiring_timer.timeout.connect(self.update_acquiring_log)
        self.acquiring_timer.start(500) #every 500ms prints a dot

        self.acquisition_timer = QTimer()
        self.acquisition_timer.setSingleShot(True)
        self.acquisition_timer.timeout.connect(self.stop_acquisition)
        self.acquisition_timer.start(minutes * 60 * 1000)

    #updates acquiring animation
    def update_acquiring_log(self):
        self.acquiring_dots = (self.acquiring_dots + 1) % 4
        dots = '.' * self.acquiring_dots
        self.log.setText(f"Acquiring{dots}")

    def stop_acquisition(self):
        if not self.serial_manager.serial or not self.serial_manager.serial.is_open:
            self.log.append("Serial not connected.")
            return
        if not self.streaming:
            return
        self.serial_manager.send_cmd([0x38,0x01,0x00])
        self.streaming = False

        if self.acquisition:
            self.acquisition = False
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            if self.acquisition_timer:
                self.acquisition_timer.stop()
                self.acquisition_timer = None
            if self.acquiring_timer:
                self.log.setText("Acquisition complete")
                self.acquiring_timer.stop()
                self.acquiring_timer = None

    def send_datetime_now(self):
        now = datetime.now()
        payload = [
            0x50, 0x06,
            now.day,
            now.month,
            now.year - 2000,
            now.hour,
            now.minute,
            now.second
        ]
        self.serial_manager.send_cmd(payload)

    #serial/parser functions
    def read_serial(self):
        if not (self.serial_manager.serial and self.serial_manager.serial.is_open):
            return
        try:
            n = self.serial_manager.serial.in_waiting
            if n:
                self.rx_buffer.extend(self.serial_manager.serial.read(n))
                self.process_buffer()
        except Exception as e:
            self.log.append(f"Read error: {e}")

    def process_buffer(self):
        while True:
            if len(self.rx_buffer) < 3:
                return
            cmd = self.rx_buffer[0]
            length = self.rx_buffer[1]
            total = 2 + length + 1
            if len(self.rx_buffer) < total:
                return
            frame = bytes(self.rx_buffer[:total])
            del self.rx_buffer[:total]  #empties buffer
            self.parser.parse_frame(frame)  #dispatches to parser

    #thread function for temp./hum. polling every 4 seconds
    def poll_temp_hum(self):
        while getattr(self, "thread_running", None):
            self.thread_req = True
            self.serial_manager.send_cmd([0x42, 0x01, 0x00])
            time.sleep(4)
