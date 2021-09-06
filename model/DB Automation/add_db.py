import pyodbc
import mysql.connector


conn = mysql.connector.connect(user='root', password='', port='3307', host='localhost', database='coviddb')


cursor = conn.cursor(buffered=True)

cursor.execute('SELECT * FROM coviddb.markers')

cursor.execute('''
                INSERT INTO coviddb.markers(id, name, address, subject, lat, lng, type)
                VALUES
('0','0','0','0','0','0','None')
                ''')

conn.commit()
