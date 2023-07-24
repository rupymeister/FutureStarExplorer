import math
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split

csv_path = "fifa_normal_attributes.csv"
df = pd.read_csv(csv_path)
df_filtered = (df[['Full Name', 'Pace Total', 'Shooting Total', 'Passing Total','Dribbling Total', 'Defending Total', 'Physicality Total', 'Position']]) 
df_filtered = df_filtered.head(1000) ## top 50 players in fifa who are not GK

print(df_filtered)
df_filtered = (df[['Position','Pace Total', 'Shooting Total', 'Passing Total','Dribbling Total', 'Defending Total', 'Physicality Total']]) 


X = df.drop('Position', axis=1)
y = df['Position']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


df_filtered = (df[['Pace Total', 'Shooting Total', 'Passing Total','Dribbling Total', 'Defending Total', 'Physicality Total']]) 
#get input
pace = int(input("Pace: ")) 
shooting = int(input("Shooting: "))
passing = int(input("Passing: "))
dribbling = int((input("Dribbling: ")))
defense = int(input("Defense: "))
physicality = int(input("Physicality: "))
k = int(input("Enter k value: "))

#turn the data into an array
df_filtered = df_filtered.head(100) ## top 50 players in fifa who are not GK
players = df_filtered.to_numpy()

print(players)

di = {}

for i, player in enumerate(players):
    for j, attribute in enumerate(player):
        paceDiff = attribute - pace
        shootingDiff = attribute - shooting
        passingDiff = attribute - passing
        dribblingDiff = attribute - dribbling
        defenseDiff = attribute - defense
        physicalityDiff = attribute - physicality
        u = math.sqrt(math.pow(paceDiff, 2) + math.pow(shootingDiff, 2) + math.pow(passingDiff, 2) + math.pow(dribblingDiff, 2) + math.pow(defenseDiff, 2) + math.pow(physicalityDiff, 2))
        di.update({i:u})

print(di)
distance = {k: v for k, v in sorted(di.items(), key=lambda item: item[1])}
print(distance)

df_position = df['Position']
df_position = df_position.head(100)
positions = df_position.to_numpy()
print("\n[Pac S Pas Dr De P]  index  position  distance")
for i in distance: 
    print(players[i], " ", i, "    ", positions[i], "     ", distance[i])




dis_list = list(distance)
print(dis_list)

k_list = []
for i in range(k):
    k_list.append(positions[dis_list[i]])

print(k_list)

attacker = defender = midfielder = 0
attacker = k_list.count("ATK")
midfielder = k_list.count("MID")
defender = k_list.count("DEF")

if attacker > defender and attacker > midfielder:
    print("Result: ATK")
elif midfielder > attacker and midfielder > defender:
    print("Result: MID")
elif defender > attacker and defender > midfielder:
    print("Result: DEF")
else:
    print("Result is unknown (Equal number of instances).")

