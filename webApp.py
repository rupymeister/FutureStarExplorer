import streamlit as st
import pandas as pd
import numpy as np



csv_path = "transfermarkt_4.csv"
df = pd.read_csv(csv_path)

df['Position'] = '-'
df.loc[(df['Best Position'] == 'ST') | (df['Best Position'] == 'LW') | (df['Best Position'] == 'RW') | (df['Best Position'] == 'RF') | (df['Best Position'] == 'LF') | (df['Best Position'] == 'CF'), 'Position'] = 'ATK'
df.loc[(df['Best Position'] == 'CDM') | (df['Best Position'] == 'CM') | (df['Best Position'] == 'CAM') | (df['Best Position'] == 'LM') | (df['Best Position'] == 'RM'), 'Position'] = 'MID' 
df.loc[(df['Best Position'] == 'LB') | (df['Best Position'] == 'CB') | (df['Best Position'] == 'RB') | (df['Best Position'] == 'LWB') | (df['Best Position'] == 'RWB'), 'Position'] = 'DEF'
df.loc[(df['Best Position'] == 'GK'), 'Position'] = 'GK'

df_filtered = df[['Full Name', 'Age', 'Overall', 'Market Value', 'Wage(in Euro)', 'Potential', 'Position', 'Positions Played', 'Best Position', 'Height(in cm)', 'Weight(in kg)', 'Preferred Foot', 'Weak Foot Rating', 
        'Skill Moves', 'Attacking Work Rate', 'Defensive Work Rate', 'Pace Total', 'Shooting Total', 'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total', 
        'Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing', 'BallControl', 'Acceleration', 'Sprint Speed', 
        'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
        'Marking','Standing Tackle','Sliding Tackle', 'Goalkeeper Diving', 'Goalkeeper Handling', ' GoalkeeperKicking', 'Goalkeeper Positioning', 'Goalkeeper Reflexes']]

work_rate_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
df_filtered['Attacking Work Rate'] = df_filtered['Attacking Work Rate'].map(work_rate_mapping)
df_filtered['Defensive Work Rate'] = df_filtered['Defensive Work Rate'].map(work_rate_mapping)


columns_to_convert = ['Wage(in Euro)', 'Height(in cm)', 'Weight(in kg)', 'Skill Moves', 'Weak Foot Rating', 'Attacking Work Rate', 'Defensive Work Rate']
df_filtered[columns_to_convert] = df_filtered[columns_to_convert].astype('int64')


def adjust_market_value(value):
    if value == '-':
        return np.nan
    elif isinstance(value, float):
        return value
    elif 'mil. €' in value:
        return float(value.replace(' mil. €', '').replace('.', '').replace(',', '')) * 1000 
    elif ' Bin €' in value:
        return float(value.replace(' Bin €', '').replace('.', '').replace(',', '')) * 1000 
    else:
        return np.nan

df_filtered['Market Value'] = df_filtered['Market Value'].apply(adjust_market_value)
mean_market_value = df_filtered.groupby(['Overall', 'Age'])['Market Value'].transform('mean')

df_filtered['Market Value'] = pd.to_numeric(df_filtered['Market Value'], errors='coerce')
df_filtered['Market Value'].fillna(mean_market_value, inplace=True)

mean_market_value = df_filtered.groupby(['Age', 'Potential'])['Market Value'].transform('mean')

df_filtered['Market Value'] = pd.to_numeric(df_filtered['Market Value'], errors='coerce')
df_filtered['Market Value'].fillna(mean_market_value, inplace=True)

mean_market_value = df_filtered.groupby(['Overall', 'Best Position'])['Market Value'].transform('mean')

df_filtered['Market Value'] = pd.to_numeric(df_filtered['Market Value'], errors='coerce')
df_filtered['Market Value'].fillna(mean_market_value, inplace=True)


# Convert market values to float
df_filtered['Market Value'] = df_filtered['Market Value'].astype('float64')

df_filtered['Market Value'].fillna(0, inplace=True)
# Convert market values to object type
df_filtered['Market Value'] = df_filtered['Market Value'].astype('int64')
df_filtered.to_csv('transfermarkt_3.csv')


market_value = df_filtered['Market Value'].to_numpy()


df_filtered = pd.read_csv('transfermarkt_3.csv', index_col= 0)

df_rising_stars = df_filtered[df_filtered['Potential'] > df_filtered['Overall']]
#df_rising_stars = df_rising_stars.loc[(df['Position'] != 'GK')]

df_experienced_wolves = df_filtered[(df_filtered['Potential'] <= df_filtered['Overall']) & (df_filtered['Overall'] >= 84)]
#df_experienced_wolves = df_experienced_wolves.loc[(df['Position'] != 'GK')]

columns_to_drop = ['Full Name', 'Age', 'Overall', 'Wage(in Euro)', 'Potential', 'Position', 'Best Position', 'Height(in cm)', 'Weight(in kg)', 'Preferred Foot', 'Weak Foot Rating', 
        'Skill Moves', 'Attacking Work Rate', 'Defensive Work Rate']
greater_than_85 = df_experienced_wolves.select_dtypes(include='int64')
greater_than_85.info()
greater_than_85 = greater_than_85.loc[:, (greater_than_85[:] >= 85).any()]
greater_than_85_array = greater_than_85.to_numpy()


df = df_filtered
df['Winger'] = '-'
df['Scoring'] = '-'
df['Defense'] = '-'
df['Offense'] = '-'
df.loc[(df['Best Position'] == 'ST') | (df['Best Position'] == 'LW') | (df['Best Position'] == 'RW') | (df['Best Position'] == 'RF') | (df['Best Position'] == 'LF') | (df['Best Position'] == 'CF') | (df['Best Position'] == 'CAM'), 'Scoring'] = '+'
df.loc[(df['Best Position'] == 'CM') | (df['Best Position'] == 'CAM') | (df['Best Position'] == 'LM') | (df['Best Position'] == 'RM') | (df['Best Position'] == 'LWB') | (df['Best Position'] == 'RWB'), 'Offense'] = '+' 
df.loc[(df['Best Position'] == 'LB') | (df['Best Position'] == 'CB') | (df['Best Position'] == 'RB') | (df['Best Position'] == 'LWB') | (df['Best Position'] == 'RWB') | (df['Best Position'] == 'CDM'), 'Defense'] = '+'
df.loc[(df['Best Position'] == 'LB') | (df['Best Position'] == 'LWB') | (df['Best Position'] == 'RB') | (df['Best Position'] == 'RWB') | (df['Best Position'] == 'LM') | (df['Best Position'] == 'RM') | (df['Best Position'] == 'RW') | (df['Best Position'] == 'LW'), 'Winger'] = '+'
df.loc[(df['Best Position'] == 'GK'), 'Player Type'] = 'GoalKeeper'


positions_played = df['Positions Played']


best_position = df['Best Position']

combined_df = pd.merge(positions_played, best_position, left_index=True, right_index=True)


combined_array = combined_df.to_numpy()


combined_df = pd.concat([df['Positions Played'], df['Best Position']], axis=1)

combined_df['Combined Positions'] = combined_df.apply(lambda x: ','.join([x['Positions Played'], x['Best Position']]), axis=1)


combined_df['Combined Positions'] = combined_df['Combined Positions'].apply(lambda x: ','.join(set(x.split(','))))
df['Combined Positions'] = combined_df['Combined Positions']

combined_positions = combined_df['Combined Positions'].to_numpy()
df_image = pd.read_csv("transfermarkt_4.csv")
df_image = df_image[['Full Name', 'Image Link', 'Market Value']]




def distance(from_df, aim_to_find, p, attributes):
    result_list = []
    count = 1
    coefficient = 2
    for _, player in from_df.iterrows():
        for _, row in aim_to_find.iterrows():
            temp_diff = np.sum(np.abs(player[attributes] - row[attributes]) ** coefficient ** p) ** (1/p)
            result_list.append(temp_diff)
        if(count % 5 == 0 & coefficient != 0):
            coefficient -= 1
        count += 1

    result_arr = np.array(result_list)
    return result_arr




def main():
    
    st.title("Player Recommendation App")
    player_names = df_experienced_wolves['Full Name'].to_numpy().tolist()
    col1, col2  = st.columns(2)
    with col1:
         name_input = st.selectbox("Select a player:", player_names)
    
    with col2:
        selected_row = df_image[df_image['Full Name'] == name_input]
        image = selected_row.iloc[0]['Image Link']
        st.image(image,  width=100)
    
    

    matching_row = df[df['Full Name'] == name_input]
    desired_positions = []
    if not matching_row.empty:
        desired_positions = matching_row['Combined Positions'].values[0].split(',')
        if "," in matching_row['Combined Positions'].values[0]:
            desired_positions += ["All"]
        
        
        position_input = st.selectbox("Specify the position:", desired_positions)
        matching_players = []
        for index, row in df.iterrows():
            combined_positions = row['Combined Positions'].split(',')
            if position_input in combined_positions:
                matching_players.append(row['Full Name'])
        matching_players = np.array(matching_players)
        
        age_input = st.number_input("Enter the maximum age:", value=20)
        market_value_input = st.number_input("Enter the maximum market value:", value=200000000)

        attributes = ['Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing', 'BallControl', 'Acceleration', 'Sprint Speed', 
        'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
        'Marking','Standing Tackle','Sliding Tackle', 'Goalkeeper Diving', 'Goalkeeper Handling', ' GoalkeeperKicking', 'Goalkeeper Positioning', 'Goalkeeper Reflexes']
        if(position_input == "All"):
            df_rising_stars = df.loc[(df['Age'] <= age_input) &
                                    (df['Height(in cm)'].between(matching_row['Height(in cm)'].values[0] - 10,
                                                                matching_row['Height(in cm)'].values[0] + 10)) &
                                    (df['Attacking Work Rate'].between(matching_row['Attacking Work Rate'].values[0] - 1,
                                                                        matching_row['Attacking Work Rate'].values[0] + 1)) &
                                    (df['Defensive Work Rate'].between(matching_row['Defensive Work Rate'].values[0] - 1,
                                                                        matching_row['Defensive Work Rate'].values[0] + 1)) &
                                    (df['Potential'] > df['Overall']) & (df['Market Value'] <= market_value_input) &
                                    (df['Skill Moves'] >= matching_row['Skill Moves'].values[0] - 2) &
                                    (df['Weak Foot Rating'] >= matching_row['Weak Foot Rating'].values[0] - 2) &
                                    (df['Overall'] <= matching_row['Overall'].values[0])]
        else:
            df_rising_stars = df.loc[(df['Age'] <= age_input) & (df['Combined Positions'].str.contains(position_input) == True) &
                                    (df['Height(in cm)'].between(matching_row['Height(in cm)'].values[0] - 10,
                                                                matching_row['Height(in cm)'].values[0] + 10)) &
                                    (df['Attacking Work Rate'].between(matching_row['Attacking Work Rate'].values[0] - 1,
                                                                        matching_row['Attacking Work Rate'].values[0] + 1)) &
                                    (df['Defensive Work Rate'].between(matching_row['Defensive Work Rate'].values[0] - 1,
                                                                        matching_row['Defensive Work Rate'].values[0] + 1)) &
                                    (df['Potential'] > df['Overall']) & (df['Market Value'] <= market_value_input) &
                                    (df['Skill Moves'] >= matching_row['Skill Moves'].values[0] - 2) &
                                    (df['Weak Foot Rating'] >= matching_row['Weak Foot Rating'].values[0] - 2) &
                                    (df['Overall'] <= matching_row['Overall'].values[0])]


        k = st.slider("Select how many to recommend (k):", min_value=1, max_value=5, value=0)
        result_arr = distance(matching_row, df_rising_stars, 2, attributes)

        sorted_indices = np.argsort(result_arr)
        sorted_distances = result_arr[sorted_indices]
    
        if sorted_indices.shape[0] == 0:
            st.warning("There are no players for that filters!")
            
        else:
            closest_index = sorted_indices[0]
            closest_name = df_rising_stars.iloc[closest_index]['Full Name']

            st.write("Recommendations for", name_input)
            count = 0
            if(k == 0):
                cols = st.columns(1)
                st.warning("There are no players for that filters!")
            else:
                cols = st.columns(k)

            for i in range(1, min(k + 1, sorted_indices.shape[0])):
                full_name = df_rising_stars.iloc[sorted_indices[i]]['Full Name']
                after = df_rising_stars.iloc[sorted_indices[i + 1]]['Full Name'] if i + 1 < sorted_indices.shape[0] else ""
                before = ""

                matching_row1 = df_image[df_image['Full Name'] == full_name]

                if not matching_row1.empty:
                    image_link = matching_row1.iloc[0]['Image Link']
                    market_val = matching_row1.iloc[0]['Market Value']
                    with cols[count % 5]:  
                        
                        st.image(image_link, caption=market_val, width=100)
                        st.write(i,full_name)

                else: 
                   
                   st.warning("There are no players for that filters!")

                count = count + 1

    else:
        st.write("Player not found according to the given input.")

if __name__ == "__main__":
    main()
