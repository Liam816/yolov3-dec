import Jetson.GPIO as GPIO
import serial
import time

# serial
ser = serial.Serial('/dev/ttyUSB0', '115200', timeout=1)
# if not ser.is_open:
#     ser.open()
# ser.flush()


serial_port = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
time.sleep(1)
print('serial init succeed.')

# # gpio
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(11, GPIO.OUT)
# GPIO.setup(12, GPIO.OUT)
# # for speed adjustment
# GPIO.setup(13, GPIO.OUT)
# GPIO.setup(15, GPIO.OUT)


def trigger_gpio():
    # for i in range(10):
    #     GPIO.output(11, GPIO.LOW)
    #     time.sleep(3)
    #     GPIO.output(11, GPIO.HIGH)
    #     time.sleep(3)

    time.sleep(1)
    # GPIO.output(11, GPIO.LOW)  # 8
    # GPIO.output(12, GPIO.LOW)  # 7
    GPIO.output(13, GPIO.LOW)  # 6
    # GPIO.output(15, GPIO.LOW)  # 5
    time.sleep(300)


def test_serial():
    ser = serial.Serial('/dev/ttyUSB0', '115200', timeout=1)
    while True:
        # read serial port --------------------------
        serial_data = ser.readline().decode('utf-8').strip()
        time.sleep(0.1)
        if serial_data == '':
            print('data none')
            ser = serial.Serial('/dev/ttyUSB0', '115200', timeout=1)
            # time.sleep(0.1)
        else:
            print('serial_data:', serial_data)
            ser.flush()
    # while True:
    #     if serial_port.inWaiting() > 0:
    #         # data = str(serial_port.read())
    #         # if data == '9':
    #         #     print('hello')
            
    #         data = serial_port.read()
    #         print(data)

    #         # data = serial_port.readline().decode('utf-8').strip()
    #         # print(data)




if __name__ == '__main__':
    # trigger_gpio()
    test_serial()
