import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulated user-item ratings matrix
data = {
    'User': ['Alice', 'Bob', 'Charlie', 'David'],
    'Item1': [5, 3, 0, 1],
    'Item2': [4, 0, 0, 1],
    'Item3': [1, 1, 0, 5],
    'Item4': [0, 1, 5, 4],
    'Item5': [0, 0, 5, 4],
}
df = pd.DataFrame(data)
df.set_index('User', inplace=True)

# Compute similarity matrix
similarity = cosine_similarity(df.fillna(0))
sim_df = pd.DataFrame(similarity, index=df.index, columns=df.index)
print("User Similarity Matrix:\n", sim_df)

# Predict rating for Charlie on Item2
charlie_ratings = df.loc['Charlie']
unrated_items = charlie_ratings[charlie_ratings == 0].index
print("\nRecommendations for Charlie:")
for item in unrated_items:
    sim_scores = []
    ratings = []
    for user in df.index:
        if df.loc[user, item] > 0:
            sim_scores.append(sim_df.loc['Charlie', user])
            ratings.append(df.loc[user, item])
    if sim_scores:
        predicted = np.dot(sim_scores, ratings) / np.sum(sim_scores)
        print(f"{item}: Predicted Rating = {predicted:.2f}")
