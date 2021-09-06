import os
import datetime
import googlemaps
from datetime import datetime

#  Hiện tại key chỉ encode 1 location / 1 ngày
gmaps = googlemaps.Client(key='AIzaSyAOqmxPHrBW3ZzFM-76gnKK29-bA7oSqfQ')

with open('../today_infolog.txt') as f:
    raw_info = f.read()
    info_paragraphs = raw_info.split('\n\n\n\n')
    for paragraph in info_paragraphs:
        slugs = paragraph.split('\n\n')
        locations = slugs[1].split('\n')
        other_info = slugs[0].split('\n')
        patient_id = other_info[0]
        date = other_info[1]
        for location in locations:
            # Geocoding an address
            geocode_result = gmaps.geocode(location)
            if "nơi ở của bệnh nhân" in location:
                address = location.split('%')[0]
                print(patient_id + "\t\t" + location + "\t\t" + geocode_result[0] + "\t\t" + geocode_result[1] + "\t\t" + address + "\t\t" + "nơi ở\n")
            else:
                print(patient_id + "\t\t" + location + "\t\t" + geocode_result[0] + "\t\t" + geocode_result[1] + "\t\t" + location + "\n")
