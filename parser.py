from utils import u16, i16, temp_c_from_raw, rh_from_raw, format_acceleration, axes_text
from utils import crc8_sum
from config import FS_TEXT, ODR_TEXT
from datetime import datetime

class FrameParser:
    def __init__(self, app):
        self.app = app  # riferimento a SerialApp per accedere a log e stato

    #parsing received frame
    def parse_frame(self, frame: bytes):
        cmd = frame[0]
        length = frame[1]
        payload = frame[2:-1]
        crc_rx = frame[-1]
        crc_ok = (crc8_sum(frame[:-1]) == crc_rx)
        crc_txt = f"0x{crc_rx:02X} - {'VALID' if crc_ok else 'INVALID'}"

        if not crc_ok:
            self.app.log.append(f"RX (CRC ERR) {frame.hex(' ')}")
            return

        #dispatch based off received frame
        if cmd == 0x35 + 0x80:
            self.handle_35_status(payload, crc_txt)
        elif cmd == 0x36 + 0x80:
            self.app.log.append(f"ACK 0x36 (setup): {crc_txt}\n------------------------")
        elif cmd == 0x37 + 0x80:
            self.stream(payload, "Stream samples", signed=True, crc_txt=crc_txt)
        elif cmd == 0x38 + 0x80:
            self.app.log.append(f"ACK 0x38 (stop stream): {crc_txt}\n------------------------")
        elif cmd == 0x39 + 0x80:
            self.handle_sample(payload, "Single sample", signed=True, crc_txt=crc_txt)
        elif cmd == 0x40 + 0x80:
            self.handle_sample_w_time(payload, "Max Acceleration (A_ms)", signed=False, crc_txt=crc_txt)
        elif cmd == 0x41 + 0x80:
            self.handle_sample(payload, "Temp Max Acceleration (A_mt)", signed=False, crc_txt=crc_txt)
        elif cmd == 0x42 + 0x80:
            self.handle_42_temp_hum(payload, crc_txt)
        elif cmd == 0x43 + 0x80:
            self.app.log.append(f"ACK 0x43 (reset max): {crc_txt}\n------------------------")
        elif cmd == 0x50 + 0x80:
            self.app.log.append(f"ACK 0x50 (set datetime): {crc_txt}\n------------------------")
        else:
            self.app.log.append(f"Unknown cmd 0x{cmd:02X} – frame: {frame.hex(' ')}")

    #cmd 0x35 handler
    def handle_35_status(self, p: bytes, crc_txt: str):
        if len(p) != 0x0F:
            self.app.log.append(f"0x35 status: payload len {len(p)} != 15")
            return
        present = "YES" if p[0] != 0 else "NO"
        t_raw = u16(p[1], p[2])
        h_raw = u16(p[3], p[4])
        fs_code = p[5]
        odr_code = p[6]
        axes = p[7]
        periodic = "ENABLED" if p[8] != 0 else "DISABLED"

        T = temp_c_from_raw(t_raw)
        RH = rh_from_raw(h_raw)
        x_max_g = u16(p[9],  p[10]) / 1000.0    #divides by 1000 since firmware transmits as int, hence we receive acc. value in mg. in (LSB,MSB) order
        y_max_g = u16(p[11], p[12]) / 1000.0
        z_max_g = u16(p[13], p[14]) / 1000.0

        self.app.current_fs_code = fs_code
        self.app.current_odr = odr_code
        self.app.current_axes = axes

        self.app.log.append(
            "=== STATUS ===\n"
            f"{'Present:':40} {present}\n"
            f"{'Temperature:':36} {T:.2f} °C\n"
            f"{'Humidity:':39} {RH:.2f} %\n"
            f"{'Full Scale:':40} {FS_TEXT.get(fs_code, f'0x{fs_code:02X}')}\n"
            f"{'Data Rate:':39} {ODR_TEXT.get(odr_code, f'0x{odr_code:02X}')}\n"
            f"{'Enabled Axes:':36} {axes_text(axes)}\n"
            f"{'Periodic Send:':36} {periodic}\n"
            f"{'X-axis max acceleration:':29} {x_max_g:.3f} g\n"
            f"{'Y-axis max acceleration:':29} {y_max_g:.3f} g\n"
            f"{'Z-axis max acceleration:':29} {z_max_g:.3f} g\n"
            f"{'CRC:':42} {crc_txt}\n"
            "------------------------"
        )

    #generic handler for acceleration samples
    def handle_sample(self, p: bytes, label: str, signed: bool, crc_txt: str):
        #expected payload: 6 bytes -> [X_LSB][X_MSB][Y_LSB][Y_MSB][Z_LSB][Z_MSB]
        if len(p) != 6:
            self.app.log.append(f"{label}: payload len {len(p)} != 6")
            return
        x_cnt = i16(p[0], p[1]) / 1000.0 if signed else u16(p[0], p[1]) / 1000.0    #A_i is signed; A_mt and A_ms are unsigned (absolute value)
        y_cnt = i16(p[2], p[3]) / 1000.0 if signed else u16(p[2], p[3]) / 1000.0
        z_cnt = i16(p[4], p[5]) / 1000.0 if signed else u16(p[4], p[5]) / 1000.0

        self.app.log.append(
            f"{label}:\n"
            f"{format_acceleration(x_cnt, y_cnt, z_cnt, self.app.current_axes)}\n"
            f"CRC: {crc_txt}\n"
            "------------------------"
        )

    #used for cmd (0x40) since firmware sends A_ms and date/time of when the last maximum was sampled
    def handle_sample_w_time(self, p: bytes, label: str, signed: bool, crc_txt: str):
        if len(p) != 12:
            self.app.log.append(f"{label}: payload len {len(p)} != 12")
            return

        x_cnt = i16(p[0], p[1]) / 1000.0 if signed else u16(p[0], p[1]) / 1000.0
        y_cnt = i16(p[2], p[3]) / 1000.0 if signed else u16(p[2], p[3]) / 1000.0
        z_cnt = i16(p[4], p[5]) / 1000.0 if signed else u16(p[4], p[5]) / 1000.0

        dd = p[6]
        mm = p[7]
        yy = 2000 + p[8]

        hh = p[9]
        min = p[10]
        sec = p[11]

        self.app.log.append(
        f"{label}:\n"
        f"{format_acceleration(x_cnt, y_cnt, z_cnt, self.app.current_axes)} ---> "
        f"last maximum (any axis) sampled at {yy:04d}-{mm:02d}-{dd:02d} {hh:02d}:{min:02d}:{sec:02d}\n"
        f"CRC: {crc_txt}\n"
        "------------------------"
        )

    #used for cmd (0x42): firmware sends current temperature and humidity
    def handle_42_temp_hum(self, p: bytes, crc_txt: str):
        #expected payload: 4 bytes -> [T_LSB][T_MSB][RH_LSB][RH_MSB]
        if len(p) != 4:
            self.app.log.append(f"0x42: payload len {len(p)} != 4")
            return

        t_raw = u16(p[0], p[1])
        h_raw = u16(p[2], p[3])
        T = temp_c_from_raw(t_raw)
        RH = rh_from_raw(h_raw)
        self.app.current_temperature = T
        self.app.current_humidity = RH

        #if it is not a thread request, print
        if not self.app.thread_req:
            self.app.log.append(
                f"{'Temperature':30}      {T:.2f} °C\n"
                f"{'Humidity':30}         {RH:.2f} %\n"
                f"{'CRC':28}              {crc_txt}\n"
                "------------------------"
            )
        self.app.thread_req = False

    #used for cmd (0x37): streaming
    def stream(self, p: bytes, label: str, signed: bool, crc_txt: str):
        #expected payload: 6 bytes -> [X_LSB][X_MSB][Y_LSB][Y_MSB][Z_LSB][Z_MSB]
        if len(p) != 6:
            self.app.log.append(f"{label}: payload len {len(p)} != 6")
            return

        x_cnt = i16(p[0], p[1]) / 1000.0 if signed else u16(p[0], p[1]) / 1000.0
        y_cnt = i16(p[2], p[3]) / 1000.0 if signed else u16(p[2], p[3]) / 1000.0
        z_cnt = i16(p[4], p[5]) / 1000.0 if signed else u16(p[4], p[5]) / 1000.0

        #prints if it is not an acquisition
        if self.app.streaming and not self.app.acquisition:
            #header + active axes
            header = ""
            values = ""

            if self.app.current_axes & 0x01:  # X
                header += f"       X"
                values += f"{x_cnt:>7.3f} g"
            if self.app.current_axes & 0x02:  # Y
                header += f"             Y"
                values += f"{y_cnt:>7.3f} g"
            if self.app.current_axes & 0x04:  # Z
                header += f"             Z"
                values += f"{z_cnt:>7.3f} g"

            text = header + "\n" + values
            self.app.log.setText(text)
            #header = f"       X             Y            Z\n"
            #self.app.log.setText(header + f"{x_cnt:7.3f}g {y_cnt:7.3f}g {z_cnt:7.3f}g")

        #if it's an acquisition saves data in csv
        if self.app.acquisition:
            temp = self.app.current_temperature
            hum = self.app.current_humidity
            now = datetime.now()
            self.app.csv_writer.writerow([
                now.date(),
                now.time(),
                f"{x_cnt:.3f}".replace('.', ','),
                f"{y_cnt:.3f}".replace('.', ','),
                f"{z_cnt:.3f}".replace('.', ','),
                f"{temp:.2f}".replace('.', ','),
                f"{hum:.2f}".replace('.', ','),
            ])
            self.app.csv_file.flush()
