# Spotify Data Analysis Project
# Author: Vinay Bollinedi
# This script analyzes a Spotify dataset to extract insights 
# about song popularity, artists, and audio features.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load and copy the dataset
data = pd.read_csv('dataset.csv')
df = data.copy()
# Display first 5 rows of the dataset
print(df.head().iloc[:, :6])
# Shape before dropping duplicates
print("Shape before dropping duplicates:", df.shape)
# Dropping unnecessary columns
df.drop(columns=["track_id", "Unnamed: 0"], inplace=True)
# Display first 5 columns
print(df.head().iloc[:, :6])   
# Dropping duplicate rows
df.drop_duplicates(inplace=True)
# Shape after dropping duplicates
print("Shape after dropping duplicates:", df.shape)
# Convert duration from milliseconds to minutes
# ====== 1. Dataset Overview ======
print("Number of Unique Artists:", df['artists'].nunique())
print("Number of Unique Genres:", df['track_genre'].nunique())
# ====== 2. Top Songs and Artists ======
top_songs = df.sort_values(by='popularity', ascending=False).head(10)
print("\nTop 10 Most Popular Songs:")
print(top_songs[['track_name', 'artists', 'popularity']].to_string(index=False))
top_artists = df.groupby('artists')['popularity'].mean().nlargest(10)
print("\nTop 10 Artists by Average Popularity:")
print(top_artists)
# ====== 3. Correlation & Energy Analysis ======
correlation = df['danceability'].corr(df['energy'])
print("\nCorrelation between Danceability and Energy:", round(correlation, 3))
loudness_corr = df['loudness'].corr(df['energy'])
print("Correlation between Loudness and Energy:", round(loudness_corr, 3))
# ====== 4. Genre-Based Insights ======
avg_popularity = df.groupby('track_genre')['popularity'].mean().nlargest(5)
print("\nTop 5 Genres by Average Popularity:")
print(avg_popularity)
danceable_genre = df.groupby('track_genre')['danceability'].mean().nlargest(1)
energetic_genre = df.groupby('track_genre')['energy'].mean().nlargest(1)
print("\nMost Danceable Genre:", danceable_genre)
print("Most Energetic Genre:", energetic_genre)
# ====== 5. Duration Analysis ======
df['duration_min'] = df['duration_ms'] / 60000
avg_duration_min=df['duration_min'].mean()
print("\nAverage Song Duration (in minutes):", round(avg_duration_min, 2))
# Visualization of Song Duration
plt.figure(figsize=(10, 6))
sns.histplot(df['duration_min'], bins=30, edgecolor='black')
plt.title("Distribution of Song Duration", fontweight='bold')
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Songs")
plt.tight_layout()
plt.show()
# ====== 6. Summary ======
print("\n===== Summary =====")
print("Total Songs:", len(df))
print("Average Popularity:", round(df['popularity'].mean(), 2))
print("Average Duration (min):", round(df['duration_min'].mean(), 2))
