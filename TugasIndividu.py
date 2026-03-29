import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

usia = np.array([[22],[35],[48],[61],[74]])
tekanan_darah = np.array([[115],[128],[140],[155],[170]])
berat_badan = np.array([[52],[68],[75],[82],[90]])

minmax = MinMaxScaler()

usia_scaled = minmax.fit_transform(usia)
td_scaled = minmax.fit_transform(tekanan_darah)

print("Hasil Min-Max Scaling (Usia):")
print(usia_scaled)

print("\nHasil Min-Max Scaling (Tekanan Darah):")
print(td_scaled)

zscore = StandardScaler()

berat_z = zscore.fit_transform(berat_badan)

print("\nHasil Z-Score (Berat Badan):")
print(berat_z)
