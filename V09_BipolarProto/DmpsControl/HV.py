import numpy as np 
from .hardware import HaukeDMA
import time
import spidev
import RPi.GPIO as GPIO
import serial
import threading


pin_sclk = 23
pin_mosi = 19
pin_sync = 24

class HVController:
    def __init__(self, HVsource="Bipolar"):
        self.spi = spidev.SpiDev()
        self.setup()

    def setup(self):
        self.spi.open(0, 0)          # because you have /dev/spidev10.0
        self.spi.max_speed_hz = 1_000_000
        self.spi.mode = 0b01
        self.spi.bits_per_word = 8

    def write_dac8551(self, code: int):
        code = max(0, min(65535, int(code)))

        data = [
            0x00,
            (code >> 8) & 0xFF,
            code & 0xFF,
        ]

        self.spi.xfer2(data)

    def cleanup(self):
        self.spi.close()
        
    def cunningham_correction(self, dp, T=293.15, P=101325, a= 1.142, b=0.558, c=0.999, test_new_lambda=False):
        if test_new_lambda:
            lambda_0 = 38.5e-9
            a = 1.996
            b = 0.975
            c = 1.746
        
        lambda_0 = 67.3e-9
        T0 = 273.15
        P0 = 101325
        lambda_air = lambda_0 * (T / T0) * (P0 / P)
        return 1 + (2 * lambda_air / dp) * (a + b * np.exp(-c * dp / (2 * lambda_air)))
        
    def voltage_from_size(self, dp_nm, Q_sh_lpm=14.0, T_C=24.0, P=101325, debug=False):
        mu = 1.81e-5    
        e = 1.602e-19    
        negative = False
        
        dma = HaukeDMA()
        r1 = dma.r1    
        r2 = dma.r2
        L = dma.L
        
        if dp_nm <= 0:
            dp_nm = np.abs(dp_nm) 
            negative = True
            
        dp = dp_nm * 1e-9       
        Q_sh = Q_sh_lpm / 60000  
        ln_r = np.log(r2 / r1)
        T_K = T_C + 273.15

        Cc = self.cunningham_correction(dp, T=T_K, P=P)

        V = (3 * mu * Q_sh * ln_r * dp) / (2 * L * e * Cc)
        
        if debug:   
            if negative:
                print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")
            else:
                print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")
            
        if negative:
            V = -V

        return V
    
    def set_voltage(self, dp, Q_sh_lpm=10.0):
        if self.hv_source == "Bipolar":
            voltage = self.voltage_from_size(dp, Q_sh_lpm=Q_sh_lpm)
            value = self.DACValue(voltage)
            
            self.write_dac8551(value)
        elif self.hv_source == "Unipolar":
            pass
        else:
            raise ValueError(f"Unknown HV source: {self.hv_source}")
        
    def zeroDac(self):
        self.write_dac8551(32705)  # 0 V
        
    
    
    


def cunningham_correction(dp, T=293.15, P=101325, a= 1.142, b=0.558, c=0.999, test_new_lambda=False):
    if test_new_lambda:
        lambda_0 = 38.5e-9
        a = 1.996
        b = 0.975
        c = 1.746
    
    lambda_0 = 67.3e-9
    T0 = 273.15
    P0 = 101325
    lambda_air = lambda_0 * (T / T0) * (P0 / P)
    return 1 + (2 * lambda_air / dp) * (a + b * np.exp(-c * dp / (2 * lambda_air)))

def voltage_from_size(dp_nm, Q_sh_lpm=14.0, T_C=24.0, P=101325, debug=False):
    mu = 1.81e-5    
    e = 1.602e-19    
    negative = False
    
    dma = HaukeDMA()
    r1 = dma.r1    
    r2 = dma.r2
    L = dma.L
    
    if dp_nm <= 0:
        dp_nm = np.abs(dp_nm) 
        negative = True
        

    dp = dp_nm * 1e-9       
    Q_sh = Q_sh_lpm / 60000  
    ln_r = np.log(r2 / r1)
    T_K = T_C + 273.15

    Cc = cunningham_correction(dp, T=T_K, P=P)

    V = (3 * mu * Q_sh * ln_r * dp) / (2 * L * e * Cc)
    
    if debug:   
        if negative:
            print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")
        else:
            print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")
        
    if negative:
        V = -V

    return V

def DACValue(value, debug= False):
    v0 = 32705
    dacValue = value * 65535 / 15000 + v0
    
    if debug:
        print(f"writeDAC called with value: {value}")
        volt = 15000*(int(dacValue)-v0)/ 65535
        print(f"Calculated DAC value: {dacValue}, which corresponds to voltage: {volt:.1f} V")
    return int(dacValue)
    
spi = spidev.SpiDev()

def setup():
    spi.open(0, 0)          # because you have /dev/spidev10.0
    spi.max_speed_hz = 1_000_000
    spi.mode = 0b01
    spi.bits_per_word = 8

def write_dac8551(code: int):
    code = max(0, min(65535, int(code)))

    data = [
        0x00,
        (code >> 8) & 0xFF,
        code & 0xFF,
    ]

    spi.xfer2(data)

def cleanup():
    spi.close()


    
def voltage_set(dp, Q_sh_lpm=10.0):
    voltage = voltage_from_size(dp, Q_sh_lpm=Q_sh_lpm)
    value = DACValue(voltage)
    
    write_dac8551(value)


def dac_code_from_size(dp, Q_sh_lpm=10.0):
    voltage = voltage_from_size(dp, Q_sh_lpm=Q_sh_lpm)
    return DACValue(voltage)

    

def test():
    for code in [0, 16384, 32768, 49152, 65535]:
        print(f"Setting DAC to code: {code}")
        write_dac8551(code)
        time.sleep(1)

def zero():
    write_dac8551(32705)  # 0 V


class SpellmanHV:
    STATUS_BITS = {
        0: "Enabled",
        1: "Fault",
        2: "Over voltage",
        3: "Over current",
        4: "Over temperature",
        5: "Supply rail out of range",
        6: "HW enable",
        7: "SW enable",
    }

    def __init__(self, port="/dev/ttyUSB0", baud=9600, max_voltage=30000.0):
        self.port = port
        self.baud = int(baud)
        self.max_voltage = float(max_voltage)
        self.lock = threading.Lock()
        self.voltage = 0.0

    def checksum(self, body):
        total = 0
        for char in body:
            total += ord(char)
        total = 0x200 - total
        total = total | 0x40
        total = total & 0x7F
        return total

    def command(self, body):
        cmd = chr(2) + body + f"{self.checksum(body):X}" + chr(10)
        with self.lock:
            ser = None
            try:
                ser = serial.Serial(
                    self.port,
                    self.baud,
                    timeout=1,
                    write_timeout=1,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                ser.write(cmd.encode("ascii"))
                ser.flush()
                return ser.readline()
            finally:
                if ser is not None:
                    time.sleep(0.1)
                    ser.close()

    def clear_faults(self):
        return self.command("0106CF=1")

    def enable(self):
        return self.command("0106EN=1")

    def disable(self):
        return self.command("0106EN=0")

    def set_voltage(self, voltage):
        voltage = max(0.0, min(self.max_voltage, abs(float(voltage))))
        body = "0106V1=" + f"{voltage:07.1f}"
        self.command(body)
        self.voltage = voltage

    def voltage_set(self, dp, Q_sh_lpm=10.0):
        self.set_voltage(voltage_from_size(abs(float(dp)), Q_sh_lpm=Q_sh_lpm))

    def zero(self):
        self.set_voltage(0.0)

    def get_voltage(self):
        try:
            res = self.command("0106M0?")
            text = res.decode("ascii", errors="ignore") if isinstance(res, bytes) else str(res)
            idx = text.find("M0=")
            if idx < 0:
                return self.voltage
            value = ""
            for char in text[idx + 3:]:
                if char in "0123456789.-":
                    value += char
                else:
                    break
            return float(value) if value else self.voltage
        except Exception:
            return self.voltage

    def get_status(self):
        res = self.command("0106SR?")
        text = res.decode("ascii", errors="ignore") if isinstance(res, bytes) else str(res)
        idx = text.find("SR=")
        if idx < 0:
            return None
        raw = text[idx + 3: idx + 7]
        if len(raw) < 4:
            return None
        value = int(raw, 16) & 0xFF
        return {
            "raw": raw,
            "value": value,
            "bits": {name: bool(value & (1 << bit)) for bit, name in self.STATUS_BITS.items()},
        }

def hv():
    write_dac8551(32000)  # -137 V
    time.sleep(1)

if __name__ == "__main__":
    setup()
    time.sleep(1)
    try:
        zero()
    finally:
        cleanup()
