import pandas as pd
import numpy as np
data = pd.read_csv('/Users/devanshkedia/Desktop/AI YOGA/pose_angles.csv')

print(data['left_knee_angle'].max())
print(data['left_knee_angle'].min())

threshold = 30
count1 = (data["left_knee_angle"] < threshold).sum()
count2 = (data["right_knee_angle"] < threshold).sum()
print(count1)
print(count2)