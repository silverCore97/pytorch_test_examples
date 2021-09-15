import sqlite3
import sys, tkinter

# ende function for getting rid of the GUI
def ende():
    main.destroy()

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

# Create Tk object
main = tkinter.Tk()

list=["n","m","epoch","accuracy","time"]
#Create top line of the table
for i in range(len(list)):
    lb = tkinter.Label(main, text=str(list[i]), bg="#FFFFFF", bd=5,
                   relief="sunken", anchor="e")
    lb.grid(row=0, column=i, sticky="we")

#Rest of the table
iterator=1
for dsatz in cursor:
    for i in range(len(list)):
        lb = tkinter.Label(main, text=str(round(dsatz[i],2)), bg="#FFFFFF", bd=5,
                   relief="sunken", anchor="e")
        lb.grid(row=iterator, column=i, sticky="we")
    iterator +=1

# Verbindung beenden
connection.close()

main.mainloop()

