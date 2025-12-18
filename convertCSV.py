import pandas as pd
import os

df = pd.read_csv(os.path.join('tabellen', 'Grundlage Sensordatenbank_V3.csv'),sep=';', encoding='utf8')
# eval_df = pd.DataFrame(columns=["Data Format", "Data raw"]) # , "Question", "Answer"
# csv_data = df.to_csv(index=False)
# eval_df.loc[len(eval_df)] = ["CSV", csv_data]





# einzelString = ''
# for nrZeile in range(len(df.values)):
#     for nrColumn in range(len(df.columns)):
#         einzelString = einzelString + df.columns[nrColumn] + ': ' + str(df.values[nrZeile][nrColumn]) + ';'
#     einzelString = einzelString + '\n\n'

einzelString = ''
for nrZeile in range(0, 114):
    zeile = ('Der Hersteller ' + str(df.values[nrZeile][1]) + ' hat den Sensor mit den Namen ' + str(df.values[nrZeile][2]) + 
                ' entwickelt. Der Sensor misst die ' + str(df.values[nrZeile][3]) + ' im Bereich von ' + str(df.values[nrZeile][5]) + 
                ' bis ' + str(df.values[nrZeile][6]) + ' ' + str(df.values[nrZeile][7]) + '. Angeschlossen wird der Sensor mit einen ' +
                str(df.values[nrZeile][8]) + '. Verwendet wird ein ' + str(df.values[nrZeile][9]) + ' Messfühler. Das Ausgangssinganl des Sensors ist ' +
                str(df.values[nrZeile][10]) + '. Der Sensor kann Temperaturen von ' + str(df.values[nrZeile][22]) + ' bis ' + str(df.values[nrZeile][23]) + ' Celsius aushalten. \n\n')
    
    einzelString = einzelString + zeile
for nrZeile in range(115, len(df.values)):
    zeile = ('Der Hersteller ' + str(df.values[nrZeile][1]) + ' hat den Sensor mit den Namen ' + str(df.values[nrZeile][2]) + 
                ' entwickelt. Der Sensor misst die ' + str(df.values[nrZeile][3]) + ' im Bereich von ' + str(df.values[nrZeile][5]) + 
                ' bis ' + str(df.values[nrZeile][6]) + ' ' + str(df.values[nrZeile][7]) + '. Der Sensor arbeitet im Bereich von ' + 
                 str(df.values[nrZeile][14]) + ' bis '  + str(df.values[nrZeile][15]) + ' ' +  str(df.values[nrZeile][16]) + 
                 '. Der Sensor hat die Zulassungen: ' + str(df.values[nrZeile][17]) + ', dazu besitz der Sensor Folgene Schutzarten: ' + str(df.values[nrZeile][21]) +
                '. Der Sensor kann Temperaturen von ' + str(df.values[nrZeile][22]) + ' bis ' + str(df.values[nrZeile][23]) + ' Celsius aushalten. \n\n')
    einzelString = einzelString + zeile
    

with open(os.path.join('usedQuellen', 'sensorTabelle.txt'),'w', encoding="utf-8") as txt_file:
        txt_file.write(einzelString)

print("Tabelle umgewandelt.")
test = 7