#crc (checksum mod 256) calculation
def crc8_sum(data_bytes):
    return sum(data_bytes) & 0xFF

#converts 2 bytes (unsigned)
def u16(lsb, msb):
    return (msb << 8) | lsb

#converts 2 bytes (signed)(two's complement)
def i16(lsb, msb):
    val = (msb << 8) | lsb
    return val - 0x10000 if val & 0x8000 else val

#used for printing the enabled axis on cmd 0x35
def axes_text(axes_mask):
    out = []
    if axes_mask & 0x01: out.append("X")
    if axes_mask & 0x02: out.append("Y")
    if axes_mask & 0x04: out.append("Z")
    return " ".join(out) if out else "—"

#used for printing
def format_acceleration(x, y, z, axes_mask):
    parts = []
    if axes_mask & 0x01:
        parts.append(f"X: {x:>7.3f} g")
    if axes_mask & 0x02:
        parts.append(f"Y: {y:>7.3f} g")
    if axes_mask & 0x04:
        parts.append(f"Z: {z:>7.3f} g")
    return " | ".join(parts) if parts else "No axis enabled"

#converts 2 bytes (based on SHT40's datasheet conversion formula) in temperature (°C)
def temp_c_from_raw(raw_u16):
    return -45.0 + 175.0 * (raw_u16 / 65535.0)

#converts 2 bytes (based on SHT40's datasheet conversion formula) in relative humidity(%)
def rh_from_raw(raw_u16):
    rh = -6.0 + 125.0 * (raw_u16 / 65535.0)
    return max(0.0, min(100.0, rh))     #normalizing in [0%,100%]
