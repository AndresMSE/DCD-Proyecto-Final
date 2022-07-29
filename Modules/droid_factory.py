import numpy as np
from pip import main
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Activation, Flatten, \
Conv3D,MaxPooling3D,Input,GlobalAveragePooling3D
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model,X_test,y_test,type_cm,display_labels,cmap=plt.cm.Blues,normalize=True):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    c_mat = confusion_matrix(y_test,y_pred)
    if normalize:
        c_mat = np.round(c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis],4)
    plt.imshow(c_mat, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix'+type_cm)
    plt.colorbar()
    tick_marks = np.arange(len(display_labels))
    plt.xticks(tick_marks, display_labels, rotation=0)
    plt.yticks(tick_marks, display_labels)
    thresh = c_mat.max() / 2.
    for i, j in itertools.product(range(c_mat.shape[0]), range(c_mat.shape[1])):
        plt.text(j, i, c_mat[i, j],
                 horizontalalignment="center",
                 color="white" if c_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def disp_confusion_matrix(model,X,y,type_cm,class_label):
    disp = plot_confusion_matrix(model, X, y, type_cm,display_labels=class_label,
                                 cmap=plt.cm.Blues)

tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
X_train = np.load('CWT-D/X_train_s.npy')
X_test = np.load('CWT-D/X_test_s.npy')
y_train = np.load('CWT-D/y_train_s.npy')
y_test = np.load('CWT-D/y_test_s.npy')


def get_model(width=38, height=300, depth=6,activation='sigmoid',dense_activation='sigmoid',dropout_rate=0.3):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    
    model.add(Input((width, height, depth, 1)))

    model.add(Conv3D(filters=19, kernel_size=3, activation=activation,padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,1)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=19, kernel_size=3, activation=activation,padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=38, kernel_size=3, activation=activation,padding='same'))
    model.add(MaxPooling3D(pool_size=(3,5,1)))
    model.add(BatchNormalization())

#     model.add(Conv3D(filters=76, kernel_size=3, activation=activation,padding='same'))
#     model.add(MaxPooling3D(pool_size=(1,5,1)))
#     model.add(BatchNormalization())

#     model.add(GlobalAveragePooling3D())
    model.add(Flatten())
    model.add(Dense(units=76, activation=dense_activation))
    model.add(Dropout(dropout_rate))

#     model.add(Dense(units=38, activation=dense_activation))
#     model.add(Dropout(dropout_rate))

#     model.add(Dense(units=256, activation=dense_activation))
#     model.add(Dropout(dropout_rate))

    model.add(Dense(units=19, activation='tanh'))
#     model.add(Dropout(dropout_rate))

    model.add(Dense(units=3,kernel_regularizer=tf.keras.regularizers.l1(l=0.0001)))
    model.add(Activation('softmax',name='CNN_EEGModel'))


    return model

def testers(dense_act,lr,channels,epochs=100,verb=1):
    model = get_model(width=38,height=300,depth=6,activation='relu',dense_activation=dense_act,
                  dropout_rate=0.3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy',optimizer=opt,
             metrics=['accuracy'])
    num_epochs = epochs
    X_test_res = X_test[:100,1:,:,channels]
    history = model.fit(X_train[:2000,1:,:,channels],y_train[:2000], epochs = num_epochs,\
                        shuffle=True, verbose=verb,validation_data=(X_test_res,y_test[:100]))
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Costo vs Época')
    #plt.xlim(0,50)
    plt.ylabel('Costo')
    plt.xlabel('Época')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    disp_confusion_matrix(model,X_test_res,y_test[100:200],' first six set A',class_label=['Left Hand','Right Hand','Neutral']) 

    if __name__== '__main__':
        comb = [4,7,9,14,15,21]
        testers('sigmoid',1e-5,comb)