import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
import keras_tuner as kt
import os.path
from sklearn.metrics import confusion_matrix
import pickle
from pathlib import Path
from functions import *

# Data Set Information:
# - Predicting forest cover type from cartographic variables only
# - Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables
#  (wilderness areas and soil types).
# - This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.
#  These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more
#  a result of ecological processes rather than forest management practices.
# - Number of Attributes: 12 measures, but 54 columns of data (10 quantitative variables, 4 binary wilderness areas
#  and 40 binary soil type variables)
# - No missing values

#################################
### 1. Prepare the data #########
#################################

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                   compression='gzip', sep=",", header=None)
X, y = data_preparation(data)
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X, y)

#################################
### 2. Simple Heuristic #########
#################################

# As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak
# would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5).
# Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4).
# Wilderness Areas:
# - 1 -- Rawah Wilderness Area
# - 2 -- Neota Wilderness Area
# - 3 -- Comanche Peak Wilderness Area
# - 4 -- Cache la Poudre Wilderness Area

# Create a simple heuristic by the description
# Neota -> 1, Rawah and Comanche Peak -> 2, and Cache la Poudre -> 3

heuristic_pred, heuristic_acc = heuristic(X,y)
# This simple heuristic was able to predict the correct Cover Type with accuracy of about 54%.

#################################
### 3. Scikit-Learn ML models ###
#################################

# 3.1 Logistic regression
log_reg = LogisticRegression(multi_class='ovr', solver='liblinear')
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_acc = log_reg.score(X_test, y_test)

# 3.2 XGBoost
xgb = XGBClassifier(n_estimators = 500, max_depth = 4, learning_rate = 0.05, random_state = 0)
xgb.fit(X_train, y_train-1)  # It needs labels from 0 to 6
xgb_pred = xgb.predict(X_test)
xgb_acc = xgb.score(X_test, y_test-1)

#################################
### 4. Neural Network ###########
#################################

# 4.1. Preprocess the data
scaled_train_samples, train_labels, scaled_valid_samples, valid_labels, scaled_test_samples, test_labels = \
    neural_network_data_preprocessing(X_train, y_train, X_valid, y_valid, X_test, y_test)

# 4.2. Build a model
tuner = kt.Hyperband(hypermodel=model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='trials')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x=scaled_train_samples, y=train_labels,
             validation_data = (scaled_valid_samples, valid_labels),
             epochs=50, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(scaled_train_samples, train_labels, epochs=50,
                    validation_data = (scaled_valid_samples, valid_labels), verbose=0)
# Save models
if os.path.isfile('models/hypermodel.h5') is False:
    hypermodel.save('models/hypermodel.h5')

MODELS_PATH = Path() / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)
with open(MODELS_PATH / "logistic_reg.pkl", 'wb') as file:
    pickle.dump(log_reg, file)
with open(MODELS_PATH / "xgb.pkl", 'wb') as file:
    pickle.dump(xgb, file)

# Save figures
IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
# Save history figures of accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
save_fig('accuracy_history', IMAGES_PATH)
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
save_fig('loss_history', IMAGES_PATH)
plt.clf()

#################################
### 5. Evaluation ###############
#################################

predictions = hypermodel.predict(x=scaled_test_samples, verbose=0)
hypermodel_pred = np.argmax(predictions, axis=-1)
hypermodel_acc = hypermodel.evaluate(scaled_test_samples, test_labels)[1]

# Save figures of confusion matrices
cm_plot_labels_1 = [0, 1, 2, 3, 4, 5, 6]  # for hypermodel and xgboost
cm_plot_labels_2 = [1, 2, 3, 4, 5, 6, 7]  # for heuristic and logistic regression
# Compute confusion matrices
cm_heuristic = confusion_matrix(y_true=y, y_pred=heuristic_pred)
cm_log_reg = confusion_matrix(y_true=y_test, y_pred=log_reg_pred)
cm_xgb = confusion_matrix(y_true=y_test-1, y_pred=xgb_pred)
cm_hypermodel = confusion_matrix(y_true=test_labels, y_pred=hypermodel_pred)
# Plot and save figures
plot_confusion_matrix(cm=cm_heuristic, classes=cm_plot_labels_2, title='Heuristic_CM',
                      figure_name='Heuristic_CM', img_path=IMAGES_PATH)
plot_confusion_matrix(cm=cm_log_reg, classes=cm_plot_labels_2, title='Log_reg_CM',
                      figure_name = 'Log_reg_CM', img_path=IMAGES_PATH)
plot_confusion_matrix(cm=cm_xgb, classes=cm_plot_labels_1, title='XGB_CM',
                      figure_name = 'XGB_CM', img_path=IMAGES_PATH)
plot_confusion_matrix(cm=cm_hypermodel, classes=cm_plot_labels_1, title='Hypermodel_CM',
                      figure_name = 'Hypermodel_CM', img_path=IMAGES_PATH)

# Save accuracies as a csv table
accuracy_table = pd.DataFrame(data={'Model':['heuristic', 'logistic regression', 'xgboost', 'hypermodel'],
                                'Accuracy': [heuristic_acc, log_reg_acc, xgb_acc, hypermodel_acc]}).set_index('Model')
accuracy_table.to_csv('accuracy.csv')
