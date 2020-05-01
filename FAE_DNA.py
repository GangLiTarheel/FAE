
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.layers import Input, Dense, Lambda, Concatenate, Reshape, RepeatVector
from keras.models import Model, Sequential
from keras.models import model_from_json


from keras.optimizers import RMSprop, Adam

import matplotlib.pyplot as plt
import sys
import os

class FAE():
	def __init__(self):
		# Input shape
		self.samples = 60000 
		self.dim = 250
		self.T =15 
		self.channels = 1
		#self.img_shape = (self.samples, self.dim, self.channels)
		self.L0=60
		self.b0=5
		
		# Build and compile the autoencoder
		#self.autoencoder = self.build_autoencoder()
		self.optimizer = Adam()#RMSprop(),


		# Initialzie input
		# Oberservation
		L_input = Input(shape = (self.dim, self.channels),name='L')
		C_input = Input(shape=(self.dim,  self.channels), name='C')

		# Hidden Variable
		U_input = Input(shape=(self.dim,  self.channels), name='U')	
		V_input = Input(shape=(self.dim,  self.channels), name='C')		

		# Constant 
		F0 = Input(shape=(self.dim,  self.channels), name='F0')
		
			
		# Build encoder
		self.encoder = self.build_encoder()
		#self.encoder.compile(loss='mean_squared_error',
		#	optimizer=self.optimizer)
		#encoder = Model(inputs=[L_input,C_input, U_input, V_input], outputs=[p1, p2, lam]])
		#mu = self.encoder([x_input, z_input])
		p1, p2, lam = self.encoder([L_input,C_input, U_input, V_input])
		
		
		# Build decoder1
		self.decoder1 = self.build_decoder1()
		#self.decoder.compile(loss='mean_squared_error',
		#	optimizer=self.optimizer)
		#self.decoder.trainable = False
		#x_hat = self.decoder([mu, z_input])
		#decoder1 = Model(inputs=[p1,p2, U_input,V_input], outputs=Lt)
		Lt = self.decoder1([F0, p1,p2, U_input])

		# Build decoder2
		self.decoder2 = self.build_decoder2()
		Ct = self.decoder2([lam,Lt,V_input])
		
		# The combined model (conect encoder and decoder)
		self.autoencoder = Model(inputs=[L_input,C_input, U_input, V_input,F0], outputs=[Lt, Ct]) # p1, p2, lam,
		self.autoencoder.compile(optimizer = self.optimizer,
		loss={'model_1': 'mean_squared_error', 
		'model_2': 'mean_squared_error'},
		loss_weights={'model_1': 1, 
		'model_2': 1})
		self.autoencoder.summary()

		# # Build the generator
		# self.generator = self.build_generator()

		# # The generator takes noise as input and generates imgs
		# z = Input(shape=(self.latent_dim,))
		# img = self.generator(z)

		# # For the combined model we will only train the generator
		# self.discriminator.trainable = False

		# # The discriminator takes generated images as input and determines validity
		# valid = self.discriminator(img)

		# # The combined model  (stacked generator and discriminator)
		# # Trains the generator to fool the discriminator
		# self.combined = Model(z, valid)
		# self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_encoder(self):
		# this is our input placeholder
		# Oberservation
		L_input = Input(shape = (self.dim, self.channels),name='L')
		C_input = Input(shape=(self.dim,  self.channels), name='C')

		# Hidden Variable
		U_input = Input(shape=(self.dim,  self.channels), name='U')	
		V_input = Input(shape=(self.dim,  self.channels), name='V')												 

		# "encoded" is the encoded representation of the input
		encoded=keras.layers.concatenate([L_input,C_input, U_input, V_input])
		encoded = Dense(32, activation='relu')(encoded)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(512, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(512, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(32, activation='relu')(encoded)
		# mu_candidate = Dense(1, activation='relu',name='mu_hat')(encoded)
		# mu = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid')(mu_candidate)
		sum_p_candidate = Dense(1, activation='sigmoid',name='sum_p_hat')(encoded)
		sum_p = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid')(sum_p_candidate)

		eta_candidate = Dense(1, activation='sigmoid',name='eta_hat')(encoded)
		eta = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid')(eta_candidate)
		
		def pp1(ip):
			sum_p = ip[0]
			eta = ip[1]
			return sum_p * eta		#mu2 + K.pow(mu2,3/2)*z2

		def pp2(ip):
			sum_p = ip[0]
			eta = ip[1]
			#mu2=K.repeat_elements(mu1,self.dim,1) #m=10 for 10 dim data
			return sum_p * (1-eta)	

		p1 = keras.layers.Lambda(pp1)([sum_p,eta])
		p2 = keras.layers.Lambda(pp2)([sum_p,eta])

		lam_candidate = Dense(1, activation='relu',name='lam_hat')(encoded)
		lam = keras.layers.AveragePooling1D(pool_size=self.dim, strides=None, padding='valid')(lam_candidate)

		encoder = Model(inputs=[L_input,C_input, U_input, V_input], outputs=[p1, p2, lam])
		
		encoder.summary()

		return encoder


	def build_decoder1(self):
		# this is our input placeholder
		# Parameters
		p1 = Input(shape = (1,  self.channels),name='p1')
		p2 = Input(shape = (1,  self.channels),name='p2')
		#lam = Input(shape = (1,  self.channels),name='lam')

		# Hidden Variable
		U_input = Input(shape=(self.dim,  self.channels), name='U')	
		#V_input = Input(shape=(self.dim,  self.channels), name='C')	

		# Constant variables
		L0=K.constant(self.L0) # initial length of sequence
		b0=K.constant(self.b0) # number of blocks (tandems)
		## Lt hat
		# p3
		def pp(ip):
			tp1 = ip[0]
			tp2 = ip[1]
			one = K.constant(value=1)	#repeat_elements([1],1,1) #m=10 for 10 dim data
			tp3 = one - tp1 - tp2
			return tp3
		p3 = keras.layers.Lambda(pp)([p1,p2])

		# Ft with self.T
		# Note: invert step CDF
		batch_size = tf.shape(U_input)[0]
		#Ft = tf.ones([batch_size, self.dim,self.channels])
		F0 = Input(shape = (self.dim,self.channels),name='F0')
		Ft = F0
		def CDF(ip):
			F00 = ip[0]
			tp1 = ip[1]
			tp2 = ip[2]
			tp3 = ip[3]
			batch_size = tf.shape(F00)[0]
			t = K.concatenate(tensors=[tf.zeros([batch_size, self.dim, self.channels]),F00,tf.ones([batch_size, self.dim, self.channels])], axis=-1)*p3 
			return t
		#	
		for i in range(self.T):
			Ft = keras.layers.Lambda(CDF)([Ft,p1,p2,p3])
		
		#U = RepeatVector(2*self.T+1)([U_input])
		def repeat(u_np):
			tU = K.repeat_elements(u_np,rep=2*self.T+1, axis=2)
			return tU
		
		U = keras.layers.Lambda(repeat)(U_input)


		def dgm_Lt(ip):             
			tU = ip[0]
			tFt = ip[1]
			tt=(K.sign(tU-tFt)+K.ones([self.dim, 2*self.T+1]))/2
			tt=K.sum(tt,axis=2)
			L0=K.constant(60) # initial length of sequence
			b0=K.constant(5) # number of blocks (tandems)
			tLt = b0 * tt + L0
			return tLt

		Lt = keras.layers.Lambda(dgm_Lt)([U,Ft])
		Lt = Reshape((self.dim, self.channels))(Lt)
		
		# def dg(ip):
		# 	mu1 = ip[0]
		# 	z2 = ip[1]
		# 	mu2=K.repeat_elements(mu1,self.dim,1) #m=10 for 10 dim data
		# 	return mu2 + K.pow(mu2,3/2)*z2
			
		# x_hat = keras.layers.Lambda(dg)([mu,z_input])
				
		decoder1 = Model(inputs=[F0,p1,p2, U_input], outputs=[Lt])
		
		decoder1.summary()
		

		return decoder1

	def build_decoder2(self):
		# this is our input placeholder
		# Parameters
		lam = Input(shape = (1,  self.channels),name='lam')
		Lt = Input(shape=(self.dim,  self.channels), name='Lt')	

		# Hidden Variable
		V_input = Input(shape=(self.dim,  self.channels), name='V')	

		# Constant variables
		L0=K.constant(self.L0) # initial length of sequence

		## Ct hat

		
		def lam_para(tLt):
			tlamt = lam*(L0+tLt)/2
			#tlamt = K.mean(tlamt)
			return tlamt
		lamt = keras.layers.Lambda(lam_para)(Lt)


		#samples = tf.random.poisson(lamt,[1])#[0.5, 1.5, 2.5], [1])
		#test = tf.math.igammac(K.constant([1,2,3]),lamt) # k+1, lambda
		def invert_poisson(ip):
			u, lamt = ip[0], ip[1]
			init = ( K.constant(0), tf.reshape(tf.math.igammac(K.constant(0)+1,lamt),[]) )
			c = lambda kk, pp: tf.greater(u, pp)
			b = lambda kk, pp: (tf.add(kk, 1) , tf.math.igammac(kk+1,lamt))
			r1 = tf.while_loop(c, b, init)#,shape_invariants=[k.get_shape(), pp.get_shape()])
			return r1[0]
		#u = K.constant(V_input)#np.random.uniform(0,1,1))#batch_size))
		#Ct = keras.layers.Lambda(invert_poisson)([V_input,lamt])	


		# def dg(ip):
		# 	mu1 = ip[0]
		# 	z2 = ip[1]
		# 	mu2=K.repeat_elements(mu1,self.dim,1) #m=10 for 10 dim data
		
		decoder2 = Model(inputs=[lam,Lt,V_input], outputs=lamt)
		
		decoder2.summary()


		return decoder2

	
	def load_data_model1(self):
		m = self.dim
		n = self.samples
		file_name = f'model1_{m}_{n}.npz'
		data_file = os.path.join("/nas/longleaf/home/franklee/FAE/Aug2019/data/",file_name)
		data = np.load(data_file)
		return data

	
	def train_AE(self, epochs, batch_size=256):
	
		data =  self.load_data_model1()
		train_x=data['train_x']
		train_z=data['train_z']
		train_z_mle=data['train_z_mle']
		train_mu_mle=data['train_mu_mle']
		valid_x=data['valid_x']
		valid_z=data['valid_z']
		valid_z_mle=data['valid_z_mle']
		valid_mu_mle=data['valid_mu_mle']
		train_mu=data['train_mu']
		valid_mu=data['valid_mu']
		r=data['r']
		
		#for epoch in range(epochs):
		#loss = .train_on_batch(noise, valid)
		self.train = self.autoencoder.fit({'x': train_x, 'z': train_z},
			{'model_2': train_x, 
			'model_1': train_mu[:,np.newaxis,np.newaxis] },
			batch_size=batch_size,epochs=epochs,verbose=1,
			validation_data=({'x': valid_x, 'z': valid_z},
							{'model_2': valid_x,
							'model_1':valid_mu[:,np.newaxis,np.newaxis] }))
								
		# evaluate the model
		scores = self.autoencoder.evaluate(x=[valid_x, valid_z],y=[valid_x,valid_mu[:,np.newaxis,np.newaxis]], verbose=0)
		print("%s: %.2f%%" % (self.autoencoder.metrics_names[0], scores[0]))
		print("%s: %.2f%%" % (self.autoencoder.metrics_names[1], scores[1]))
		# Plot the progress
		# print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)) 
		
		# If at save interval => save generated image samples
		# if epoch % save_interval == 0:
		# self.save_imgs(epoch)
		
		
	def save_model(self):
		# serialize model to JSON
		model_json = self.autoencoder.to_json()
		with open("FAE_DNA.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.autoencoder.save_weights("FAE_DNA.h5")
		print("Saved model to disk")
		
		# r, c = 5, 5
		# noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		# gen_imgs = self.generator.predict(noise)

		# # Rescale images 0 - 1
		# gen_imgs = 0.5 * gen_imgs + 0.5

		# fig, axs = plt.subplots(r, c)
		# cnt = 0
		# for i in range(r):
			# for j in range(c):
				# axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				# axs[i,j].axis('off')
				# cnt += 1
		# fig.savefig("images/mnist_%d.png" % epoch)
		# plt.close()


if __name__ == '__main__':
	fae = FAE()
	#fae.train_AE(epochs=5, batch_size=250)#(epochs=4000, batch_size=32, save_interval=50)
	#fae.save_model()
