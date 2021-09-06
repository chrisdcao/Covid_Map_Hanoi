import os
import datetime

patient_title = "Mã bệnh nhân:%"
location_title = "Địa điểm:%"
subject_title = "Độ nguy hiểm:%"

storage_file = open("db_add_info.txt", "a")

identity = 0

if os.path.exists('./last_identity.txt'):
    with open('last_identity.txt', 'r') as f:
        identity = int(f.read()[0])

with open('extract_toa_do.txt', 'r') as f:
    try:
        whole_input = f.read()
        lines = whole_input.split('\n')
        for line in lines:
            print(line)
            info_array = line.split('\t\t')
            patient_id = patient_title + info_array[0]
            location = location_title + info_array[1].replace(",", "-")
            lat = str(info_array[2])
            lng = str(info_array[3])
            try:
                if info_array[4]:
                    subject = subject_title + "Cao"
                    type = "red"
            except:
                subject = subject_title + "TB"
                type = "yellow"
            full_info = "('" + str(identity) + "','" + patient_id + "','" + location + "','" + subject + "','" + lat + "','" + lng + "'," + "'" + type + "'),\n"
            storage_file.write(full_info)
            identity += 1
    except:
        pass

storage_file.close()

with open('last_identity.txt', 'w') as f:
    f.write(str(identity))
