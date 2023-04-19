def data_preparation(data):
    # Column names:
    quantitative_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                         'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                         'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    wilderness_cols = []
    for i in range(1, 5):
        wilderness_cols += ['Wilderness_Area' + str(i)]
    soil_cols = []
    for i in range(1, 41):
        soil_cols += ['Soil_Type' + str(i)]
    qualitative_cols = wilderness_cols + soil_cols
    col_names = quantitative_cols + qualitative_cols
    # Set names
    data.columns = col_names + ['Cover_Type']
    # Select labels
    y = data['Cover_Type']
    # Drop labels
    X = data.drop(['Cover_Type'], axis=1)
    return X, y


def train_valid_test_split(X, y):
    # Split data in the train, validation, and test sets (0.70 : 0.15 : 0.15)
    from sklearn.model_selection import train_test_split
    X_train, X_reminding, y_train, y_reminding = train_test_split(X, y, train_size=0.7, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_reminding, y_reminding, train_size=0.5, random_state=0)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def heuristic(X, y):
    import numpy as np
    weights = np.array([2, 1, 2, 3])
    wa_data = X.loc[:, 'Wilderness_Area1':'Wilderness_Area4'].to_numpy()
    y_pred_heuristic = np.sum(wa_data*weights, axis=1)
    return y_pred_heuristic, np.mean(y_pred_heuristic == y)


def neural_network_data_preprocessing(X_train, y_train, X_valid, y_valid, X_test, y_test):
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler

    # Transform to numpy array and shift labels from [1,7] to [0,6] for output layer
    train_samples, train_labels = X_train.to_numpy(), y_train.to_numpy()-1
    valid_samples, valid_labels = X_valid.to_numpy(), y_valid.to_numpy()-1
    test_samples, test_labels = X_test.to_numpy(), y_test.to_numpy()-1

    train_samples, train_labels = shuffle(train_samples, train_labels, random_state=0)
    valid_samples, valid_labels = shuffle(valid_samples, valid_labels, random_state=0)
    test_samples, test_labels = shuffle(test_samples, test_labels, random_state=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(train_samples)
    scaled_valid_samples = scaler.transform(valid_samples)
    scaled_test_labels = scaler.transform(test_samples)
    return scaled_train_samples, train_labels, scaled_valid_samples, valid_labels, scaled_test_labels, test_labels


def model_builder(hp):
    from tensorflow import keras
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    # Add next layers
    model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def save_fig(fig_id, IMAGES_PATH, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, figure_name, img_path,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import numpy as np
    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_fig(figure_name, img_path)
    plt.clf()