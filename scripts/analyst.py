import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib 
from  sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from tensorflow.keras import layers, models

data_path = 'datas/'

benign_files = [f for f in os.listdir(data_path) if 'benign' in f.lower() and f.endswith('.csv')]

def prepare_training_data(file_list):
    li = []
    for f in file_list:
        df = pd.read_csv(os.path.join(data_path,f))
        li.append(df)
    
    full_df = pd.concat(li, axis=0, ignore_index=True)

    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)

    if 'label' in full_df.columns:
        full_df = full_df.drop(columns=['label'])
    
    if 'ts' in full_df.columns:
        full_df = full_df.drop(columns=['ts'])
    
    return full_df

print("Preparing training data...")
X_train_raw = prepare_training_data(benign_files)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
corr_matrix = X_train_raw.corr().abs()

threshold = 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns).drop(columns=to_drop).values

print(f"Dropped {len(to_drop)} highly correlated features")
print(to_drop)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False,cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

joblib.dump(scaler, 'scaler.gz')

print(f"Training set prepared {X_train_scaled.shape[0]} rows, {X_train_scaled.shape[1]} columns.")

available_columns = X_train_raw.drop(columns=to_drop).columns.tolist()
print(f"\n--- Available Columns ({len(available_columns)} columns) ---")

protected_columns = ['fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number','ack_flag_number'
                     ,'ece_flag_number','cwr_flag_number','HTTP','HTTPS','DNS','SSH','UDP']

cols_to_check = [col for col in X_train_raw.columns if col not in to_drop and col not in protected_columns]

selector = VarianceThreshold(threshold=0.0001)
X_to_filter = pd.DataFrame(X_train_scaled, columns=X_train_raw.drop(columns=to_drop).columns)[cols_to_check]
selector.fit(X_to_filter)

remaining_cols = [col for col, support in zip(cols_to_check, selector.get_support()) if support]
final_feature_list = protected_columns + remaining_cols

X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_raw.drop(columns=to_drop).columns)[final_feature_list].values

print(f"\n--- Protected and Distinctive ({len(final_feature_list)} columns) ---")
print(final_feature_list)
print(f"Final Data Set: {X_train_final.shape[0]} rows, {X_train_final.shape[1]} columns.")


# -------------------------------------------------

X_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.drop(columns=to_drop).columns)
X_train_final = X_df[final_feature_list].values

#  Autoencoder Mimarisi (26 -> 18 -> 9 -> 18 -> 26)
input_dim = X_train_final.shape[1]
autoencoder = models.Sequential([
    # Encoder
    layers.Input(shape=(input_dim,)),
    layers.Dense(18, activation='relu'),
    layers.Dense(9, activation='relu'), 
    
    # Decoder
    layers.Dense(18, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# 3. Model Eğitimi (İP5)
print(f"\n--- {X_train_final.shape[1]} feature education start---")
history = autoencoder.fit(
    X_train_final, X_train_final,
    epochs=10,
    batch_size=512,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# 4. Modeli ve Parametreleri Sakla
autoencoder.save('iot_model.keras')
joblib.dump(final_feature_list, 'final_features.gz')

print("\nEducation Complete! 'iot_model.keras' and 'final_features.gz' created.")