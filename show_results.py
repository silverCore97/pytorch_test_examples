import sqlite3

# Verbindung, Cursor
connection = sqlite3.connect("ML_data.db")
cursor = connection.cursor()

# SQL-Abfrage
sql = "SELECT * FROM Data"

# Kontrollausgabe der SQL-Abfrage
# print(sql)

# Absenden der SQL-Abfrage
# Empfang des Ergebnisses
cursor.execute(sql)

# Ausgabe des Ergebnisses
print("n","m","epoch","accuracy","time")
for dsatz in cursor:
    print(dsatz[0], dsatz[1], dsatz[2],
          round(dsatz[3],2), round(dsatz[4],2))

# Verbindung beenden
connection.close()

