# %% First cell
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt

# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# %%
import uproot
import pandas as pd

pathtoyourrootfile1 = '/home/alwin/Downloads/JetNtuple_RunIISummer16_13TeV_MC_103(1).root'
pathtoyourrootfile2 = '/home/alwin/Downloads/JetNtuple_RunIISummer16_13TeV_MC_88.root'
filelist1 = [pathtoyourrootfile1]
filelist2 = [pathtoyourrootfile2]
# %%
all_dfs = []  # Initialize an empty list to store DataFrames

# Process filelist1
for fl in filelist1:
    with uproot.open(fl) as file:
        thisdf = file["AK4jets/jetTree;14"].arrays(library="pd")
        all_dfs.append(thisdf)
# %%
# Process filelist2
for fl in filelist2:
    with uproot.open(fl) as file:
        thisdf = file["AK4jets/jetTree;25"].arrays(library="pd")
        all_dfs.append(thisdf)
# %%
# Concatenate all DataFrames at once
df = pd.DataFrame()
df = pd.concat(all_dfs, ignore_index=False)
# %%
pathtoyourrootfile1 = '/home/alwin/Downloads/JetNtuple_RunIISummer16_13TeV_MC_103(1).root' ## add the path of your root file in this.
pathtoyourrootfile2 = '/home/alwin/Downloads/JetNtuple_RunIISummer16_13TeV_MC_88.root'
filelist1 = [pathtoyourrootfile1]
filelist2 = [pathtoyourrootfile2]

for fl in filelist2:
    file = uproot.open(fl)
    thisdf = file["AK4jets/jetTree;26"].arrays(library="pd")
    df = pd.concat([df,thisdf])
# %%
df.head()
df = df[(df.isPhysG==1) | (df.isPhysUDS==1)].reset_index()
# %%
light_quark = df['isPhysUDS'] # for one hot encoding
gluon = df['isPhysG']
x_new = df[['QG_ptD','QG_axis2','QG_mult']]
x1_new = x_new
y_new = np.array(light_quark)
y1_new = y_new.reshape(-1)
x1_new = np.array(x1_new)
from sklearn.model_selection import train_test_split
X_trainnew, X_testnew, y_trainnew, y_testnew = train_test_split(x1_new, y1_new, test_size=0.2, random_state=42)



# %%
# 6 parameter input
x = df[['QG_ptD','QG_axis2','QG_mult','jetMass','jetGirth','jetArea']]
x1 = x
y = np.array(light_quark)
y1 = y.reshape(-1)
x1 = np.array(x1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)
# %%

x_final = df[['QG_ptD','QG_axis2','QG_mult','jetMass','jetGirth','jetArea','jetChargedHadronMult','jetNeutralHadronMult','jetChargedMult','jetNeutralMult','jetMult']]
x1_final = x_final
y_final = np.array(light_quark)
y1_final = y_final.reshape(-1)
x1_final = np.array(x1_final)
from sklearn.model_selection import train_test_split
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(x1_final, y1_final, test_size=0.2, random_state=42)
# %%
from tensorflow import keras
from tensorflow.keras import layers, Model

# Define input
inputs = keras.Input(shape=(X_train_final.shape[1],))

# Hidden layers
x = layers.Dense(64, activation='relu', name="hidden_1")(inputs)
x = layers.Dense(64, activation='relu', name="hidden_2")(x)

# Output layer
outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

# Build the model                              
model_final = keras.Model(inputs=inputs, outputs=outputs)

# Compile
model_final.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Train
history_final = model_final.fit(
    X_train_final, y_train_final,
    epochs=100,
    batch_size=100,
    validation_split=0.3
)

# Now you can extract features safely
feature_extractor = Model(inputs=model_final.input,
                          outputs=model_final.get_layer("hidden_2").output)

# Extract features for test data
extracted_features = feature_extractor.predict(X_test_final)

# Access training history
training_loss_final = history_final.history['loss']
training_accuracy_final = history_final.history['accuracy']
validation_loss_final = history_final.history['val_loss']
validation_accuracy_final = history_final.history['val_accuracy']
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extracted_features is (N, 64)
# Convert to DataFrame for convenience
df_features = pd.DataFrame(extracted_features, columns=[f"neuron_{i+1}" for i in range(extracted_features.shape[1])])

# Compute correlation matrix
corr_matrix = df_features.corr()

# Show correlation matrix
print(corr_matrix)

# Optional: visualize
plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix of Extracted Features")
plt.show()

# %%
# Get model predictions (probabilities between 0 and 1 for binary classification)
import seaborn as sns
y_pred_probs = model_final.predict(X_test_final)

# Make sure predictions are a 1D array
y_pred_probs = y_pred_probs.flatten()
# Make a DataFrame for only the test features
x_test_df = pd.DataFrame(X_test_final, columns=x_final.columns)

# Add predictions as a new column
x_test_df["NN_prediction"] = y_pred_probs

# Compute correlations
corr_matrix = x_test_df.corr()

# Extract correlations with NN_prediction
corr_with_pred = corr_matrix["NN_prediction"].drop("NN_prediction")

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,         # show values in the cells
    fmt=".2f",          # 2 decimal places
    cmap="coolwarm",    # red-blue color scheme
    center=0            # white at 0 correlation
)
plt.title("Correlation Matrix (Features + NN Prediction)", fontsize=14)
plt.show()

# %%
epochs = range(1, len(training_loss_final) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss_final, 'b', label='Training Loss')
plt.plot(epochs, validation_loss_final, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
min_var_loss = min(validation_loss_final)
min_var_ind = validation_loss_final.index(min_var_loss)
print("Minimum loss:" , min_var_loss)
print("Index:", min_var_ind)
# %%
