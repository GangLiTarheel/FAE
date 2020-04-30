
#from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np


from keras.layers import Input, Dense, Lambda, Concatenate, Reshape
from scipy.stats import poisson

mu=Input([1,10])




#p3=1-p1-p2

# print(p1)
# K.get_value(p1)
#import keras.backend as K

# tf.print(p1)#, output_stream=sys.stdout)#message='p1 = ')
# tf.print(p2)#, output_stream=sys.stdout)#message='p2 = ')

constants = [1,2,3]
k_constants = K.variable(constants)
fixed_input = Input(tensor=k_constants)
print(fixed_input)
K.get_value(fixed_input)

x = tf.constant([[1.0, 2.0], [3.0, 4.0]]) 
# tf.keras.backend.print_tensor(x) 
sess = tf.Session()
with sess.as_default():
    p1=K.constant(value=0.02)#Input([1])
    p2=K.constant(value=0.04)#Input([1])
    def pp(ip):
        tp1 = ip[0]
        tp2 = ip[1]
        one=K.constant(value=1)#repeat_elements([1],1,1) #m=10 for 10 dim data
        return one - tp1 - tp2
    p3 = keras.layers.Lambda(pp)([p1,p2])
    #mu=Input([1,10])
    batch_size = 3#tf.shape(mu)[0]
    T = 5
    u_np = K.constant(np.random.uniform(0,1,batch_size))
    U = K.repeat_elements(u_np,rep=2*T+1, axis=0)
    U = K.reshape(U, shape=(batch_size, 2*T+1))#, input_shape=(batch_size*(2*T+1),))(U)
    Ft = tf.ones([batch_size, 1])
    def CDF(ip):
        F00 = ip[0]
        tp1 = ip[1]
        tp2 = ip[2]
        tp3 = ip[3]
        #one=K.constant(value=1)
        #zero=K.constant(value=0)
        batch_size = tf.shape(F00)[0]
        #padding_tensor = tf.ones([batch_size, 1])
        t = K.concatenate(tensors=[tf.zeros([batch_size, 1]),F00,tf.ones([batch_size, 1])], axis=-1)*p3 + K.concatenate(tensors=[F00,tf.ones([batch_size, 1]),tf.ones([batch_size, 1])], axis=-1)*p1 + K.concatenate(tensors=[tf.zeros([batch_size, 1]),tf.zeros([batch_size, 1]),F00], axis=-1)*p2
        return t
    for i in range(T):
        Ft = keras.layers.Lambda(CDF)([Ft,p1,p2,p3])
    tt=(K.sign(U-Ft)+tf.ones([batch_size, 2*T+1]))/2
    tt=K.sum(tt,axis=1)
    tt=K.constant(3)
    L0=K.constant(60) # initial length of sequence
    b0=K.constant(5) # number of blocks (tandems)
    Lt = tt*b0+L0
    lam = K.constant(np.random.uniform(0,0.005,batch_size))#0.005 
    lamt = lam*(L0+Lt)/2
    lamt = K.mean(lamt)
    #samples = tf.random.poisson(lamt,[1])#[0.5, 1.5, 2.5], [1])
    test = tf.math.igammac(K.constant([1,2,3]),lamt) # k+1, lambda
    def invert_poisson(ip):
        u, lamt = ip[0], ip[1]
        init = ( K.constant(0), tf.reshape(tf.math.igammac(K.constant(0)+1,lamt),[]) )
        c = lambda kk, pp: tf.greater(u, pp)
        b = lambda kk, pp: (tf.add(kk, 1) , tf.math.igammac(kk+1,lamt))
        r1 = tf.while_loop(c, b, init)#,shape_invariants=[k.get_shape(), pp.get_shape()])
        return r1[0]
    u = K.constant(0.84)#np.random.uniform(0,1,1))#batch_size))
    k = keras.layers.Lambda(invert_poisson)([u,lamt])

    # N=7
    # for k in range(7):
    #     p_pmf = K.exp(-lamt) * K.pow(lamt,k) / 
    #rv = poisson(lam)
    #poi_cdf = rv.cdf(range(5))
    # def nrandom_poisson(p, lam):
    #     rv = poisson(lam)
    #     var, pp = 0, rv.pmf(0)
    #     while(p>pp):
    #         var=var+1
    #         pp=pp+rv.pmf(var)
    #     return var
    tensor = k
    print_op = tf.print(tensor)
    with tf.control_dependencies([print_op]):
        out = tf.add(tensor, tensor)
    sess.run(out)
    #sess.run(tf.exp(tf.lgamma(5.0)))

# tf.print(fixed_input)#, output_stream=sys.stdout)#message='fixed_input = ')
# with tf.Session() as sess:
#     print(fixed_input.eval()) 
