import RPi.GPIO as GPIO
import os
import numpy as np
import time
from picamera2 import Picamera2
import piexif
from PIL import Image
import cv2 
from datetime import datetime
from dronekit import connect, VehicleMode
from ultralytics import YOLO

image_no=0
count = 0
scrapper  = YOLO("scrapper.pt")
shape_counter = YOLO("shape_counter.pt")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size": (1920, 1080)}, main={"format":'RGB888', "size": (1920, 1080)}))

picam2.start()

path = "/home/asunama/missions"

# Init GPIO, use GPIO pin 5 as input trigger from APM/Pixhawk
camTrigger = 5
GPIO.setmode(GPIO.BCM) # specify which mode to use(BCM/BOard) for pin numbering
GPIO.setwarnings(False) # It is possible that you have more than one script/circuit on the GPIO of your Raspberry Pi. As a result of this, 
                        # if RPi.GPIO detects that a pin has been configured to something other than the default (input), you get a warning when you try to configure a script. To disable these warnings:
GPIO.setup(camTrigger, GPIO.IN, GPIO.PUD_UP) # You need to set up every channel you are using as an input or an output. To configure a channel as an input:GPIO.setup(channel, GPIO.IN)
                                             # pull_up (bool or None) – If True (the default), the GPIO pin will be pulled high by default. In this case, connect the other side of the button to ground. If False, the GPIO pin will be pulled low by default. In this case, connect the other side of the button to 3V3. If None, the pin will be floating

def write_mission_no(mission_no):
    with open(f"{path}/.mission", "w") as f:
        f.write(str(mission_no))
        
def get_mission_no():
    if(os.path.exists(f"{path}.mission")):
        with open(f"{path}/.mission", "r") as f:
            n = f.read()
            if len(n)==0:
                return 1
            else:
                return int(n)
    else:
        write_mission_no(1)
        return 1



def apply_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(sobelx*2 + sobely*2)
    
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    return gradient_magnitude

def second_prediction(cropped_img):
    class_names = shape_counter.model.names
    results2 = shape_counter.predict(cropped_img)
    cnt = {}
    for r in results2: 
        for box in r.boxes:
            clss = int(box.cls)
            if class_names[clss] not in cnt:
                cnt[class_names[clss]] = 1
            else:
                cnt[class_names[clss]] += 1
    return cnt

def predict(img):
    results1 = scrapper.predict(img)
    tot_res = []
    for result in results1:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int,box.xyxy[0])

            cropped_img = img[y1:y2, x1:x2]
            cropped_img = apply_filter(cropped_img)
            tot_res.append(second_prediction(cropped_img))
    return tot_res


def get_date_time():
    return datetime.now().strftime("%y-%m-%d__%H:%M:%S")


def convert_to_dms(coordinate):
    """Convert a floating-point GPS coordinate into DMS format used in EXIF."""
    degrees = int(coordinate)
    minutes = int((coordinate - degrees) * 60)
    seconds = int(((coordinate - degrees) * 60 - minutes) * 60 * 10000)  # Seconds scaled by 10000 for EXIF
    return degrees, minutes, seconds

def add_gps_to_image(lat, lon):
    # image = Image.fromarray(img_array)

    # Convert coordinates to DMS for EXIF
    lat_dms = convert_to_dms(abs(lat))
    lon_dms = convert_to_dms(abs(lon))

    # Prepare GPS info for EXIF
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if lat >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude: ((lat_dms[0], 1), (lat_dms[1], 1), (lat_dms[2], 10000)),
        piexif.GPSIFD.GPSLongitudeRef: b'E' if lon >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude: ((lon_dms[0], 1), (lon_dms[1], 1), (lon_dms[2], 10000)),
    }

    # Insert GPS info into EXIF data
    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes

    # Save the image with EXIF data
    image.save(f"{path}/{output_path}", exif=exif_bytes)
    if(count<=6):
        image.save(f"{path}/output/{output_path}", exif=exif_bytes)
    print(f"GPS coordinates added to {output_path}")

def capture(vehicle, mission_no, dtime, count):
    global image_no
    while True:
        if GPIO.input(camTrigger) == True: # reading input (high/low) 
            count+=1
            print("Shutter Triggered")
            location = vehicle.location.global_relative_frame
            img = picam2.capture_array()
            os.makedirs(f"{path}/mission_{mission_no}_{dtime}", exist_ok=True)
            os.makedirs(f"{path}/output/mission_{mission_no}_{dtime}", exist_ok=True)
            # results = predict(img)
            # if(results is not None):
            #     pass
            exif = add_gps_to_image(img, f"{path}/mission_{mission_no}_{dtime}/image{image_no+1}.jpg", location.lat, location.lon, count)
            image_no += 1
            print(location.lat, location.lon)

try:
    mission_no = get_mission_no()
    write_mission_no(mission_no+1)
    dtime = get_date_time()
    vehicle = connect("127.0.0.1:14551", wait_ready=True)
    print("Vehicle connected successfully")
    capture(vehicle, mission_no, dtime, count)
except Exception as e:
    print(e)
finally:
    with open("home/asunama/missions/mission_{mission_no}_{dtime}/count.txt", "w") as f:
        f.write(f"{count} hotspots captured" )
    GPIO.cleanup()
    picam2.stop()
    picam2.close()
