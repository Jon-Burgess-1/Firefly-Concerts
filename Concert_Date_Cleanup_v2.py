#%% Packages
###Create models to predict the amount of beer to order for concerts based on historical sales data
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from dotenv import load_dotenv
from datetime import datetime, time, timedelta

#%% Change WD
os.chdir(r'C:\Python Projects\Firefly')



#%% Load in APIs and Keys from env
load_dotenv()
g_key = os.getenv("g_key")

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")



#%% Load each sheet from the google doc with Google API
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Load credentials
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet by ID or name
sheet = client.open_by_key(g_key)

worksheet23 = sheet.worksheet('LIVE 2023')
worksheet24 = sheet.worksheet('LIVE 2024')
worksheet25 = sheet.worksheet('LIVE 2025')

# Get data as list of dicts
data_23 = worksheet23.get_all_records()
data_24 = worksheet24.get_all_records()
data_25 = worksheet25.get_all_records()

# Convert to DataFrame
df_23 = pd.DataFrame(data_23)
df_24 = pd.DataFrame(data_24)
df_25 = pd.DataFrame(data_25)

#Convert the blank spaces "" as default from google docs to NAN
df_23.replace("", np.nan, inplace=True)
df_24.replace("", np.nan, inplace=True)
df_25.replace("", np.nan, inplace=True)

#%% Clean the data for 2023

'''
2023 is the most straightforward but has some cancelled shows to remove. Will also need to account
for the change in beer from 2023 to 2025 at the end
'''
#Look at data types and count nan
df_23.isna().sum()
df_23.dtypes


#Remove last row with the sum value
df_23 = df_23.iloc[:-1]

#Remove shows that were cancelled or moved
df_23 = df_23.dropna(thresh=df_23.shape[1] - 2)

#Split DATE column into Day of Week and Date
df_23[['Day', 'Date']] = df_23['DATE'].str.split(', ', n=1, expand=True)

#Drop original DATE column
df_23 = df_23.drop(columns="DATE")

#Convert Date to datetime object
df_23['Date'] = pd.to_datetime(df_23['Date'])

#Get weekday number
df_23['Weekday_Num'] = df_23['Date'].dt.weekday

#Convert door and showtime to datetime
df_23['DOORS'] = pd.to_datetime(df_23['DOORS'], format='%I:%M %p').dt.time
df_23['SHOWTIME'] = pd.to_datetime(df_23['SHOWTIME'], format='%I:%M %p').dt.time

#Clean up EST PPL column and convert to numeric
df_23['EST PPL'] = df_23['EST PPL'].astype(str)


#%% Clean the data for 2024
#Look at data types and count nan
df_24.isna().sum()
df_24.dtypes

#Remove all rows after the 14th/Keep the first 14
df_24 = df_24.iloc[:15]

#Split DATE column into Day of Week and Date
df_24[['Day', 'Date']] = df_24['DATE'].str.split(', ', n=1, expand=True)

#Drop original DATE column
df_24 = df_24.drop(columns="DATE")

#Convert Date to datetime object
df_24['Date'] = pd.to_datetime(df_24['Date'])

#Get weekday number
df_24['Weekday_Num'] = df_24['Date'].dt.weekday

#Convert DOORS and SHOWTIME to datetime 
df_24['DOORS_dt'] = pd.to_datetime(df_24['DOORS'], format='%I%M%p')
df_24['SHOWTIME_dt'] = pd.to_datetime(df_24['SHOWTIME'], format='%I%M%p')

#Extract just the time if needed
df_24['DOORS'] = df_24['DOORS_dt'].dt.time
df_24['SHOWTIME'] = df_24['SHOWTIME_dt'].dt.time

#Drop the temporary datetime columns from two above
df_24 = df_24.drop(['DOORS_dt', 'SHOWTIME_dt'], axis = 1)


#%% Clean 2025
'''
This section will probably need to be revisited throughout the concert season as shows are added
to make sure that the data entered is the same
'''
#Look at data types and count nan
df_25.isna().sum()
df_25.dtypes

#Drop rows after 18/keep firsrt 20
df_25 = df_25.iloc[:21]

# Drop concerts that have not occured and thus no data
df_25 = df_25.dropna(thresh=df_23.shape[1] - 5)

#Split DATE column into Day of Week and Date
df_25[['Day', 'Date']] = df_25['DATE'].str.split(', ', n=1, expand=True)

#Drop original DATE column
df_25 = df_25.drop(columns="DATE")

#Convert Date to datetime object
df_25['Date'] = pd.to_datetime(df_25['Date'])

#Get weekday number
df_25['Weekday_Num'] = df_25['Date'].dt.weekday

#Convert DOORS and SHOWTIME to datetime 
df_25['DOORS_dt'] = pd.to_datetime(df_25['DOORS'], format='%I%M%p')
df_25['SHOWTIME_dt'] = pd.to_datetime(df_25['SHOWTIME'], format='%I%M%p')

#Extract just the time if needed
df_25['DOORS'] = df_25['DOORS_dt'].dt.time
df_25['SHOWTIME'] = df_25['SHOWTIME_dt'].dt.time

#Work on the PIT? column
df_25.rename(columns={'PIT?': 'PIT'}, inplace=True)
# Extract numbers using regex
df_25['Pit_Count'] = df_25['PIT'].str.extract(r'(\d+)')

# Convert to numeric (int), set invalid entries as NaN
df_25['Pit_Count'] = pd.to_numeric(df_25['Pit_Count'], errors='coerce')



#Drop the temporary datetime columns and other random
df_25 = df_25.drop(['DOORS_dt', 'SHOWTIME_dt', 'Guiness', 'PIT'], axis = 1)



#%% Combine all
# Adjust 2023 column names
df_23 = df_23.rename(columns={'EST PPL': 'GA TIX', 'Soda/Mocktail': 'Soda_Mocktail', 'White Claw' : 'Seltzer'})

#Adjust 2024 column names
df_24 = df_24.rename(columns={'Soda/Mocktail': 'Soda_Mocktail', 'White Claw' : 'Seltzer', 'Modelo esp': 'Modelo'})


'''
2025 has the swap from white claws to ranch waters so need to combine into one seltzer
'''
#Adjust 2025 column names
df_25 = df_25.rename(columns={'Soda/Mocktail': 'Soda_Mocktail', 'LR ranch water' : 'RW', 'Modelo esp': 'Modelo'})
# Combine white claws and ranch water
df_25['Seltzer'] = df_25.fillna(0)['White Claw'] + df_25.fillna(0)['RW']

#Merge

merged_df = pd.concat([df_23, df_24, df_25], ignore_index=True, sort=False)


# Change merged column names that will become problematic
merged_df = merged_df.rename(columns={'Vodka Cocktail': 'Vodka_Cocktail', 'GA TIX' : 'GA', 'Holy City': 'Holy_City', 'N/A':'NA', 'EVENT':'ARTIST'})
merged_df['GA'] = pd.to_numeric(merged_df['GA'], errors= 'coerce')


#%% Get genre for EVENT with spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

def add_artist_metadata(df, artist_col='ARTIST', client_id='', client_secret=''):
    # Authenticate with Spotify
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    current_year = datetime.now().year

    def get_all_albums(artist_id):
        """Fetches all albums/singles for an artist with pagination."""
        albums = []
        limit = 50
        offset = 0

        while True:
            result = sp.artist_albums(
                artist_id=artist_id,
                album_type='album,single',
                limit=limit,
                offset=offset
            )
            items = result.get('items', [])
            if not items:
                break
            albums.extend(items)
            offset += limit
        return albums

    def get_artist_info(artist_name):
        try:
            result = sp.search(q=artist_name, type='artist', limit=5)
            items = result['artists']['items']
            if not items:
                # Added None for the two new metrics
                return ('Unknown', None, None, None)

            # Prefer exact match if possible
            for item in items:
                if item['name'].lower() == artist_name.lower():
                    artist = item
                    break
            else:
                artist = items[0]

            genres = ', '.join(artist['genres']) if artist['genres'] else 'Unknown'
            artist_id = artist['id']
            
            # --- NEW: Extract Popularity and Followers ---
            popularity = artist.get('popularity', None)
            followers = artist.get('followers', {}).get('total', None)
            # ---------------------------------------------

            # Fetch all albums/singles to estimate debut year
            albums = get_all_albums(artist_id)
            release_years = []

            for album in albums:
                try:
                    release_date = album['release_date']
                    year = int(release_date[:4])
                    release_years.append(year)
                except:
                    continue

            debut_year = min(release_years) if release_years else None
            years_active = current_year - debut_year if debut_year else None

            # Return the new metrics in the tuple
            return (genres, years_active, popularity, followers)

        except Exception as e:
            print(f"Error processing '{artist_name}': {e}")
            return ('Unknown', None, None, None)

    # Apply the metadata retrieval function
    results = df[artist_col].apply(get_artist_info)
    
    df['GENRE'] = results.apply(lambda x: x[0])
    df['YEARS_ACTIVE'] = results.apply(lambda x: x[1])
    # --- NEW: Map the new metrics to DataFrame columns ---
    df['SPOTIFY_POPULARITY'] = results.apply(lambda x: x[2])
    df['SPOTIFY_FOLLOWERS'] = results.apply(lambda x: x[3])
    # -----------------------------------------------------

    return df



merged_df = add_artist_metadata(
    merged_df,
    client_id= client_id,
    client_secret=client_secret
)

'''
CREATE A NEW GENRE COLUMN WITH MANUAL VALUE SINCE SPOTIFY HAS BLANKS AND IS AMBIGUOUS
'''
# Set to list so I can make sure names are spelled correctly and visualize better
artist_list = merged_df['ARTIST'].tolist()
print(artist_list)

#Blank Dict
manual_genre = {}

#All the artists
manual_genre['My Morning Jacket'] = "Alt/Indie"
manual_genre['Fleet Foxes'] = "Alt/Indie"
manual_genre['Two Friends'] = 'Pop'
manual_genre['Dirtyheads'] = "Reggae"
manual_genre['Trampled by Turtles/Shakey Graves'] = 'Folk'
manual_genre['Gregory Alan Isakov'] = 'Folk'
manual_genre['Willie Nelson'] = "Country"
manual_genre['JJ Grey & Mofro'] = 'Rock'
manual_genre['Social Distortion & Bad Religion'] = 'Rock'
manual_genre['Queens of the Stone Age'] = 'Rock'
manual_genre['Chappell Roan'] = 'Pop'
manual_genre['Pixies & Modest Mouse'] = 'Alt/Indie'
manual_genre['T-Pain'] = 'Hip-Hop'
manual_genre['Iration w/Pepper'] = 'Reggae'
manual_genre['311 w/Awolnation & Neon Trees'] = "Rock"
manual_genre['Band of Horses/ City & Colour'] = 'Alt/Indie'
manual_genre['Sublime w/Rome'] = 'Reggae'
manual_genre['Mt. Joy'] = 'Alt/Indie'
manual_genre['Ray LaMontagne + Gregory Alan Isakov'] = 'Folk'
manual_genre['Juvenile + Tiny Desk'] = 'Hip-Hop'
manual_genre['Lake Street Dive'] = 'Alt/Indie'
manual_genre['Brett Young'] = 'Country'
manual_genre['Dropkick Murphys'] = "Rock"
manual_genre['Khruangbin'] = 'Alt/Indie'
manual_genre['Gavin Adcock'] = 'Country'
manual_genre['Coheed And Cambria & Mastodon'] = "Rock"
manual_genre['Vampire Weekend'] = 'Alt/Indie'
manual_genre['Dispatch'] = 'Alt/Indie'
manual_genre['Wallows'] = 'Alt/Indie'
manual_genre['The Driver Era'] = 'Alt/Indie'
manual_genre['Cypress Hill & Atmosphere'] = 'Hip-Hop'
manual_genre['Sierra Ferrell'] = 'Country'
manual_genre['Vance Joy'] = 'Alt/Indie'
manual_genre['Sombr'] = 'Alt/Indie'
manual_genre['Young the Giant'] = 'Rock'
manual_genre['Alex Warren'] = 'Pop'
manual_genre['Rainbow Kitten Surprise'] = 'Alt/Indie'
manual_genre['Ryan Bingham'] = 'Country'


# Map values and key from manual dict
merged_df['Generic_Genre'] = merged_df['ARTIST'].map(manual_genre)







#%% Get the high temperature for the event day
merged_df.dtypes

merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')


latitude = 32.8707
longitude = -79.9829

import requests

def get_high_temp_open_meteo(date, lat, lon):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date,
        'end_date': date,
        'daily': 'temperature_2m_max',
        'timezone': 'auto'
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data['daily']['temperature_2m_max'][0]
    else:
        print(f"Error for {date}: {response.status_code}")
        return None


merged_df['High_Temp'] = merged_df['Date'].apply(lambda d: get_high_temp_open_meteo(d, latitude, longitude))
merged_df['High_Temp'] = (merged_df['High_Temp'] * 9/5) + 32



#%% Get Rain Amounts for the Day
# Function to get rain total for one date & location
# Function to get daily rain total for a given date & location
# Get the date range

merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%Y-%m-%d')

start_date = merged_df['Date'].min().strftime("%Y-%m-%d")
end_date = merged_df['Date'].max().strftime("%Y-%m-%d")

# Use archive endpoint for historical data
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={latitude}&longitude={longitude}"
    f"&daily=precipitation_sum"
    f"&start_date={start_date}&end_date={end_date}"
    "&timezone=auto"
)
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Build DataFrame from API result
rain_data = pd.DataFrame({
    'Date': pd.to_datetime(data['daily']['time']),
    'rain_total_mm': data['daily']['precipitation_sum']
})

# Merge rain totals into original DataFrame
merged_df = merged_df.merge(rain_data, on='Date', how='left')

# Optional: convert Date back to string
merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

print(merged_df)


merged_df['rain_total_in'] = merged_df['rain_total_mm'] / 25.4
merged_df = merged_df.drop(columns=['rain_total_mm'])

# Create a 'Rain_Intensity' category instead of raw inches
def categorize_rain(inches):
    if inches == 0:
        return 'None'
    elif inches < 0.2:
        return 'Light'
    else:
        return 'Heavy'

df['Rain_Intensity'] = df['rain_total_in'].apply(categorize_rain)

# This gives the model 3 clear 'buckets' to learn from
#%% Calculate the lengeth of the show (22:00 - Gates)
merged_df.dtypes
# Convert GATE to datetime.time
merged_df['DOORS'] = pd.to_datetime(merged_df['DOORS'], format='%H:%M:%S').dt.time

# Target time: 10 PM
target_time = pd.to_datetime('22:00:00', format='%H:%M:%S').time()

# Function to calculate time difference in hours
def time_diff_hours(gate_time):
    gate_dt = pd.to_datetime(str(gate_time), format='%H:%M:%S')
    target_dt = pd.to_datetime(str(target_time), format='%H:%M:%S')
    diff = target_dt - gate_dt
    return diff.total_seconds() / 3600  # hours

merged_df['Show_Duration'] = merged_df['DOORS'].apply(time_diff_hours)




#%% Create a df that has only relevant beer items and removes the other stuff


# 1. Create the Season feature
# Converting Date to datetime and mapping months to seasons
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

merged_df['Season'] = merged_df['Date'].dt.month.map(get_season)

# 2. Add Availability Flags 
# This helps the model understand that 'NaN' means 'Not Offered' rather than 'Zero Sales'
merged_df['has_montucky'] = merged_df['Montucky'].notnull().astype(int)
merged_df['has_pacifico'] = merged_df['Pacifico'].notnull().astype(int)

# 3. Define columns to drop
# We drop other beverage sales (leakage), post-show metrics (DROP), 
# and high-cardinality strings (ARTIST/GENRE)
cols_to_drop = [
    'Vodka_Cocktail', 'Whiskey ', 'ORO', 'Water', 'Soda_Mocktail', 'NA', 
    'RW', 'Agave', 'Wine', 'White Claw', 'PremB/S', 'CLUB', 'VIP ', 
    'DROP', 'ARTIST', 'GENRE', 'Pit_Count', 'DOORS', 'SHOWTIME', 'Date','Pacifico_Available', 'Montucky_Available'
]

# Create the final modeling DataFrame
Beer_Model_DF = merged_df.drop(columns=cols_to_drop)

# Save the finalized dataset for modeling
Beer_Model_DF.to_csv('Beer_Model_DF.csv', index=False)

print("Beer_Model_DF created successfully.")
print(f"Final Feature Count: {len(Beer_Model_DF.columns)}")



#%% Export back to google sheet
from gspread_dataframe import set_with_dataframe


#Comment out since the sheet has been made, any updates from here on will need different code

# Authenticate with Google Sheets
gc = gspread.service_account(filename='credentials.json')

# Open the Google Sheet by name or URL
spreadsheet = gc.open("COPY of FF Concert Tracker")

# Create a new sheet (tab) — name it something unique
worksheet = spreadsheet.add_worksheet(title="Combined_Concerts", rows=merged_df.shape[0]+1, cols=merged_df.shape[1])

# Write DataFrame to the new sheet
set_with_dataframe(worksheet, merged_df)



worksheet = spreadsheet.worksheet("Combined_Concerts")
worksheet.clear()  # optional: clear existing contents
set_with_dataframe(worksheet, merged_df)







# %% Export as csv for backup
merged_df.to_csv('merged_concert_data.csv', index=False)    


# %%
