# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:54:26 2022

@author: Rolando Quispe Mamani
"""
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

tf.random.set_seed(22)
def capa_codificador(capa_entrada,num_filtros,aplicar_batchnormalization,reducir):
  x=tf.keras.layers.Conv2D(num_filtros, kernel_size=(4,4), padding='same', strides=reducir, use_bias=False)(capa_entrada)
  if(aplicar_batchnormalization==True):
      x=tf.keras.layers.BatchNormalization()(x)#Manter con media cercana a 0 y desviacion estandar cercana a 1
  x=tf.keras.layers.LeakyReLU()(x)
  return x

def capa_decodificador(capa_entrada,num_filtros,skip_conection, aplicar_drop_out,aplicar_batchnormalization,ampliar):
  x=tf.keras.layers.Conv2DTranspose(num_filtros, kernel_size=4, strides=ampliar, padding='same', use_bias=False)(capa_entrada)
  if(aplicar_batchnormalization==True):
      x=tf.keras.layers.BatchNormalization()(x)
  if(aplicar_drop_out==True):
      x=tf.keras.layers.Dropout(0.3)(x)#Elemento regularizador para desactivar aleatoriamente conexiones
  x=tf.keras.layers.ReLU()(x)
  x=tf.keras.layers.Concatenate()([x,skip_conection])
  return x

def funcion_de_perdida_red1(val_verdadero, val_predicho):
    #Error cuadrático medio
    return tf.reduce_mean(tf.square(tf.subtract(val_verdadero,val_predicho)))

def funcion_de_perdida_red2(val_verdadero, val_predicho):
    #Error cuadrático medio
    return tf.reduce_mean(tf.square(tf.subtract(val_verdadero,val_predicho)))

def funcion_de_perdida_red_combinada(val_verdadero, val_predicho):
    #Error cuadrático medio
    return tf.reduce_mean(tf.square(tf.subtract(val_verdadero,val_predicho)))

def RedAutoencoderEB():
    e0=layers.Input(shape=(128, 128,1))

    e1 = capa_codificador(e0,72,aplicar_batchnormalization=False,reducir=2)#Ent 128x128, Sal 64x64
    e2 = capa_codificador(e1,80,aplicar_batchnormalization=True,reducir=2)#Ent 64x64, Sal 32x32
    e3 = capa_codificador(e2,96,aplicar_batchnormalization=True,reducir=2) #Ent 32x32, Sal 16x16
    e4 = capa_codificador(e3,112,aplicar_batchnormalization=True,reducir=2)#Ent 16x16, Sal 8x8
    e5 = capa_codificador(e4,128,aplicar_batchnormalization=True,reducir=2)#Ent 8x8, Sal 4x4
    e6 = capa_codificador(e5,160,aplicar_batchnormalization=True,reducir=2)#Ent 4x4, Sal 2x2
    e7 = capa_codificador(e6,192,aplicar_batchnormalization=True,reducir=2)#Ent 2x2, Sal 1x1

    e8 = capa_decodificador(e7,192,e6,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 1x1, Sal 2x2
    e9 = capa_decodificador(e8,160,e5,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 2x2, Sal 4x4
    e10 = capa_decodificador(e9,128,e4,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 4x4, Sal 8x8
    e11 = capa_decodificador(e10,112,e3,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 8x8, Sal 16x16
    e12 = capa_decodificador(e11,96,e2,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 16x16, Sal 32x32
    e13 = capa_decodificador(e12,80,e1,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 32x32, Sal 64x64
    e14=layers.Conv2DTranspose(72, kernel_size=4, strides=2, activation='relu', padding='same')(e13)#Ent 64x64, Sal 128x128
    ef=layers.Conv2D(1, kernel_size=(4, 4), activation='tanh', padding='same')(e14)

    return Model(e0,ef)

def RedAutoencoder():
    e0=layers.Input(shape=(128, 128,2))

    e1 = capa_codificador(e0,72,aplicar_batchnormalization=False,reducir=2)#Ent 128x128, Sal 64x64
    e2 = capa_codificador(e1,80,aplicar_batchnormalization=True,reducir=2)#Ent 64x64, Sal 32x32
    e3 = capa_codificador(e2,96,aplicar_batchnormalization=True,reducir=2) #Ent 32x32, Sal 16x16
    e4 = capa_codificador(e3,112,aplicar_batchnormalization=True,reducir=2)#Ent 16x16, Sal 8x8
    e5 = capa_codificador(e4,128,aplicar_batchnormalization=True,reducir=2)#Ent 8x8, Sal 4x4
    e6 = capa_codificador(e5,160,aplicar_batchnormalization=True,reducir=2)#Ent 4x4, Sal 2x2
    e7 = capa_codificador(e6,192,aplicar_batchnormalization=True,reducir=2)#Ent 2x2, Sal 1x1

    e8 = capa_decodificador(e7,192,e6,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 1x1, Sal 2x2
    e9 = capa_decodificador(e8,160,e5,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 2x2, Sal 4x4
    e10 = capa_decodificador(e9,128,e4,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 4x4, Sal 8x8
    e11 = capa_decodificador(e10,112,e3,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 8x8, Sal 16x16
    e12 = capa_decodificador(e11,96,e2,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 16x16, Sal 32x32
    e13 = capa_decodificador(e12,80,e1,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 32x32, Sal 64x64
    e14=layers.Conv2DTranspose(72, kernel_size=4, strides=2, activation='relu', padding='same')(e13)#Ent 64x64, Sal 128x128
    ef=layers.Conv2D(1, kernel_size=(4, 4), activation='tanh', padding='same')(e14)

    return Model(e0,ef)

def RedCombinada(g_model,d_model):
    for layer in d_model.layers:
        #La capa de BatchNormalization es especial, porque permite actualizar la media
        #y varianza de la entrada durante el entrenamiento.
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    e0 = layers.Input(shape=(128, 128,1))
    salida_red1 = g_model(e0)
    #print(type(salida_red1))
    #print(salida_red1.shape)
    #print(type(e0))
    #print(e0.shape)
    union=tf.concat([e0,salida_red1],-1)
    #print(type(union))
    #print(union.shape)
    salida_d = d_model(union)
    #salida_d = d_model(salida_red1)
    #print(type(salida_d))
    #print(salida_d.shape)
    return Model(e0, [salida_red1,salida_d])

def RedAutoencoderEB_RGB():
    e0=layers.Input(shape=(128, 128,3))

    e1 = capa_codificador(e0,72,aplicar_batchnormalization=False,reducir=2)#Ent 128x128, Sal 64x64
    e2 = capa_codificador(e1,80,aplicar_batchnormalization=True,reducir=2)#Ent 64x64, Sal 32x32
    e3 = capa_codificador(e2,96,aplicar_batchnormalization=True,reducir=2) #Ent 32x32, Sal 16x16
    e4 = capa_codificador(e3,112,aplicar_batchnormalization=True,reducir=2)#Ent 16x16, Sal 8x8
    e5 = capa_codificador(e4,128,aplicar_batchnormalization=True,reducir=2)#Ent 8x8, Sal 4x4
    e6 = capa_codificador(e5,160,aplicar_batchnormalization=True,reducir=2)#Ent 4x4, Sal 2x2
    e7 = capa_codificador(e6,192,aplicar_batchnormalization=True,reducir=2)#Ent 2x2, Sal 1x1

    e8 = capa_decodificador(e7,192,e6,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 1x1, Sal 2x2
    e9 = capa_decodificador(e8,160,e5,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 2x2, Sal 4x4
    e10 = capa_decodificador(e9,128,e4,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 4x4, Sal 8x8
    e11 = capa_decodificador(e10,112,e3,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 8x8, Sal 16x16
    e12 = capa_decodificador(e11,96,e2,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 16x16, Sal 32x32
    e13 = capa_decodificador(e12,80,e1,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 32x32, Sal 64x64
    e14=layers.Conv2DTranspose(72, kernel_size=4, strides=2, activation='relu', padding='same')(e13)#Ent 64x64, Sal 128x128
    ef=layers.Conv2D(3, kernel_size=(4, 4), activation='tanh', padding='same')(e14)

    return Model(e0,ef)

def RedAutoencoder_RGB():
    e0=layers.Input(shape=(128, 128,6))

    e1 = capa_codificador(e0,72,aplicar_batchnormalization=False,reducir=2)#Ent 128x128, Sal 64x64
    e2 = capa_codificador(e1,80,aplicar_batchnormalization=True,reducir=2)#Ent 64x64, Sal 32x32
    e3 = capa_codificador(e2,96,aplicar_batchnormalization=True,reducir=2) #Ent 32x32, Sal 16x16
    e4 = capa_codificador(e3,112,aplicar_batchnormalization=True,reducir=2)#Ent 16x16, Sal 8x8
    e5 = capa_codificador(e4,128,aplicar_batchnormalization=True,reducir=2)#Ent 8x8, Sal 4x4
    e6 = capa_codificador(e5,160,aplicar_batchnormalization=True,reducir=2)#Ent 4x4, Sal 2x2
    e7 = capa_codificador(e6,192,aplicar_batchnormalization=True,reducir=2)#Ent 2x2, Sal 1x1

    e8 = capa_decodificador(e7,192,e6,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 1x1, Sal 2x2
    e9 = capa_decodificador(e8,160,e5,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 2x2, Sal 4x4
    e10 = capa_decodificador(e9,128,e4,aplicar_drop_out=True,aplicar_batchnormalization=True,ampliar=2)#Ent 4x4, Sal 8x8
    e11 = capa_decodificador(e10,112,e3,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 8x8, Sal 16x16
    e12 = capa_decodificador(e11,96,e2,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 16x16, Sal 32x32
    e13 = capa_decodificador(e12,80,e1,aplicar_drop_out=False,aplicar_batchnormalization=True,ampliar=2)#Ent 32x32, Sal 64x64
    e14=layers.Conv2DTranspose(72, kernel_size=4, strides=2, activation='relu', padding='same')(e13)#Ent 64x64, Sal 128x128
    ef=layers.Conv2D(3, kernel_size=(4, 4), activation='tanh', padding='same')(e14)

    return Model(e0,ef)

def RedCombinada_RGB(g_model,d_model):
    for layer in d_model.layers:
        #La capa de BatchNormalization es especial, porque permite actualizar la media
        #y varianza de la entrada durante el entrenamiento.
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    e0 = layers.Input(shape=(128, 128,3))
    salida_red1 = g_model(e0)
    #print(type(salida_red1))
    #print(salida_red1.shape)
    #print(type(e0))
    #print(e0.shape)
    union=tf.concat([e0,salida_red1],-1)
    #print(type(union))
    #print(union.shape)
    salida_d = d_model(union)
    #salida_d = d_model(salida_red1)
    #print(type(salida_d))
    #print(salida_d.shape)
    return Model(e0, [salida_red1,salida_d])

