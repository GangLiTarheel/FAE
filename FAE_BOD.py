import keras
import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.models import model_from_json

from keras.optimizers import RMSprop, Adam

import matplotlib.pyplot as plt
import sys
import os

class FAE():
	def __init__(self):
		# Input shape
		self.samples = 120000 
		self.noise_var = 0.015        
		self.dim = 5
		self.channels = 1
		

		self.optimizer = Adam()

		# Initialzie input
		x_input = Input(shape=(self.dim, self.channels), name='x')
		y_input = Input(shape=(self.dim, self.channels), name='y')
		z_input = Input(shape=(self.dim, self.channels), name='z')
			
		# Build encoder
		self.encoder = self.build_encoder()
		self.encoder0 = self.build_encoder0()
		self.encoder1 = self.build_encoder1()
		code = self.encoder([x_input, y_input, z_input])
		t0 = self.encoder0([code])
		t1 = self.encoder1([code])
			
		# Build decoder
		self.decoder = self.build_decoder()
		y_hat = self.decoder([t0, t1, x_input, z_input])
		
		
		# The combined model (conect encoder and decoder)
		self.autoencoder = Model(inputs=[x_input,y_input,z_input], outputs=[y_hat,t0,t1])
		self.autoencoder.compile(optimizer = self.optimizer,
		loss={'decoder': 'mean_squared_error', 
		'encoder0': 'mean_squared_error',
		'encoder1': 'mean_squared_error'},
		loss_weights={'decoder': 5, 
		'encoder0': 10,
		'encoder1': 3})
		self.autoencoder.summary()


	def build_encoder(self):
		# this is our input placeholder
		x_input = Input(shape = (self.dim, self.channels),name='x')
		y_input = Input(shape = (self.dim, self.channels),name='y')
		z_input = Input(shape=(self.dim,  self.channels), name='z')
																 

		# "encoded" is the encoded representation of the input
		encoded=keras.layers.concatenate([x_input,y_input,z_input])
		encoded = Dense(32, activation='relu')(encoded)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(512, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(512, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(32, activation='relu')(encoded)
		
		encoder = Model(inputs=[x_input,y_input,z_input], outputs=encoded, name="encoder")
		
		encoder.summary()
		
		return encoder
    
	def build_encoder0(self):
		encoded0 = Input(shape=(self.dim,  32), name='code0')
		encoded = Dense(128, activation='relu')(encoded0)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(32, activation='relu')(encoded)
		t0_candidate = Dense(1, activation='relu',name='t0_hat')(encoded)
		et0 = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid',name='et0')(t0_candidate)
		encoder0 = Model(inputs=encoded0, outputs=et0, name="encoder0")
		encoder0.summary()
		return encoder0

	def build_encoder1(self):
		encoded = Input(shape=(self.dim,  32), name='code1')
		t1_candidate = Dense(1, activation='relu',name='t1_hat')(encoded)
		et1 = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid',name='et1')(t1_candidate)
		encoder1 = Model(inputs=encoded, outputs=et1, name="encoder1")
		encoder1.summary()
		return encoder1
        

	def build_decoder(self):
		# this is our input placeholder
		dt0 = Input(shape = (1,  self.channels),name='dt0')
		dt1 = Input(shape = (1,  self.channels),name='dt1')
		x_input = Input(shape = (self.dim,  self.channels),name='x')
		z_input = Input(shape=(self.dim, self.channels), name='z')
		def dg(ip):
			tt0 = ip[0]
			tt1 = ip[1]
			tx2 = ip[2]
			tz2 = ip[3]
			ty = tt0 * (1-K.exp(-tt1*tx2)) + tz2
			return ty

		y_hat = keras.layers.Lambda(dg)([dt0, dt1, x_input, z_input])
				
		encoder = Model(inputs=[dt0,dt1, x_input, z_input], outputs=y_hat, name="decoder")
		
		encoder.summary()

		return encoder


	def load_data_BOD(self,sort = True):
		m = self.dim
		n = self.samples
		x=np.array([2.0, 4.0, 6.0, 8.0, 10.0])
#m=5

		def DataGenerateFunction(z,t0,t1,m,n):
			y = z + np.diag(t0) @ (1 - np.exp(- (t1.reshape(n,1) @ x.reshape((1,m))) ) )
			return y

		def model1(m = 5, n = 30):
			t0 = np.random.uniform(0.4, 1.2, n) 
			t1 = np.random.uniform(0.000001, 0.2, n) 
			z = np.random.normal(0, self.noise_var, (n,m))
			y = DataGenerateFunction(z,t0,t1,m,n)
			return (m, n, y, z, t0, t1)

		np.random.seed(20200521)
		(m, n, y, z, t0, t1) = model1(m = m, n = n)

		if sort == True:
			p = y.argsort(axis=1)
			y.sort(axis=1)
			z=np.array([z[i,p[i,]] for i in range(n)])
		y=y[:,:,np.newaxis]
		z=z[:,:,np.newaxis]
		t0=t0[:,np.newaxis,np.newaxis]
		t1=t1[:,np.newaxis,np.newaxis]

		r=0.8 # ratio of train and validation

		train_y=y[0:int(n*r),:,:]
		train_z=z[0:int(n*r),:,:]
		valid_y=y[int(n*r):n,:,:]
		valid_z=z[int(n*r):n,:,:]

		train_t0=t0[0:int(n*r),:,:]
		train_t1=t1[0:int(n*r),:,:]
		valid_t0=t0[int(n*r):n,:,:]
		valid_t1=t1[int(n*r):n,:,:]
        
		x = np.tile(x,(n,1))
		x = x[:,:,np.newaxis]       
		train_x = x[0:int(n*r),:,:]
		valid_x = x[int(n*r):n,:,:]
                                   
		print(train_y.shape[0], 'train samples')
		print(valid_y.shape[0], 'test samples')

		return (train_x,train_y,train_z,train_t0,train_t1,valid_x,valid_y,valid_z,valid_t0,valid_t1,r)


	def train_AE(self, epochs, batch_size=256, sort = True):
        
		(train_x,train_y,train_z,train_t0,train_t1,valid_x,valid_y,valid_z,valid_t0,valid_t1,r) =  self.load_data_BOD(sort = True)
		
		self.train = self.autoencoder.fit({'x': train_x, 'y': train_y, 'z': train_z},
			{'decoder': train_y, 
			'encoder0': train_t0,
			'encoder1': train_t1},           
			batch_size=batch_size,epochs=epochs,verbose=1,
			validation_data=({'x': valid_x, 'y': valid_y, 'z': valid_z},
							{'decoder': valid_y,
							'encoder0': valid_t0,
							'encoder1': valid_t1}
							))  
        
		pred_train = self.autoencoder.predict_on_batch({'x': train_x, 'y': train_y, 'z': train_z})
		pred_valid = self.autoencoder.predict_on_batch({'x': valid_x, 'y': valid_y, 'z': valid_z})
		# np.savez('pred_train_valid.npz', pred_train=pred_train, pred_valid=pred_valid, 
        #          train_y=train_y,train_t0=train_t0,train_t1=train_t1)

		# evaluate the model
		scores = self.autoencoder.evaluate(x=[valid_x, valid_y, valid_z],y=[valid_y,valid_t0,valid_t1], verbose=0)
		print("%s: %.2f" % (self.autoencoder.metrics_names[0], scores[0]))
		print("%s: %.2f" % (self.autoencoder.metrics_names[1], scores[1]))
		print("%s: %.2f" % (self.autoencoder.metrics_names[2], scores[2]))
		print("%s: %.2f" % (self.autoencoder.metrics_names[3], scores[3]))
        
	def test_AE(self, t0=0.8, t1=1.0, n_test=1000, sort = True):
		m = self.dim
		x=np.array([2.0, 4.0, 6.0, 8.0, 10.0])


		def model_test():
			z = np.random.normal(0, self.noise_var, (1,m))
			y = z + t0 * (1 - np.exp(- (t1  * x ) ) )
			return (m, n_test, y, z, t0, t1)

		np.random.seed(20200526) # 0504
		(m, n_test, y_test, z_test, t0, t1) = model_test()
        
		y_test = np.array([0.1522071, 0.29667172, 0.41254479, 0.48237946, 0.56707723])
		p_test = np.array([0,1,2,3,4])
		print(y_test)
		print(p_test)
		y_test = np.tile(y_test,(n_test,1))
		y_test = y_test[:,:,np.newaxis]

		np.random.seed(20200504)
		z_test = np.random.normal(0, self.noise_var, (n_test,m))
		if sort == True:
			z_test=np.array([z_test[i,p_test] for i in range(n_test)])
		z_test = z_test[:,:,np.newaxis]
        
		x_test = np.tile(x,(n_test,1))
		x_test = x_test[:,:,np.newaxis]
        
		pred = self.autoencoder.predict_on_batch({'x': x_test, 'y': y_test, 'z': z_test})
		t0_fae = pred[1]
		t1_fae = pred[2]

		np.save("BOD_t0_%.3f.npy"%t0, t0_fae)
		np.save("BOD_t1_%.3f.npy"%t1, t1_fae)
		np.save("BOD_y_hat.npy", pred[0])
        

if __name__ == '__main__':
	fae = FAE()
	fae.train_AE(epochs=10, batch_size=500, sort = False) 
	fae.test_AE(t0=0.9, t1=0.1, n_test=10000, sort = False) 
