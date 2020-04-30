########################
# DNA data
# Gang Li
# 01/22/2020
#########################


import numpy as np
import operator as op
from functools import reduce
import math

## n choose r
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#block=np.tile(np.random.random_integers(0,1,4),8)

#x_0=np.concatenate((np.random.random_integers(0,1,14),block,np.random.random_integers(0,1,8)))

p1=0.2
p2=0.3

#m=10
#print(np.random.multinomial(3, [p1,p2,1-p1-p2], size=1))

T=15
def custom_random(T,p1,p2):
    p=np.zeros(2*T+1)
    for i in range(2*T+1):
        #print(i)
        if i<T: 
            #print(ncr(T,i))
            k=T-i
            # i=T-k; k=T-i
            if k==T:
                p[i]= math.pow(p1,k)
            # elif k==(T-1):
            #     p[i]= 6 * math.pow(p1,T-1) * (1-p1-p2)
            else:
                for m in range(int(math.floor((T-k)/2+1))):
                    p[i]=p[i]+ ncr(T,2*m+k) * ncr(2*m+k, m) * math.pow(p1,m+k) * math.pow(p2,m) * math.pow((1-p1-p2),(T-2*m-k))
                #p[2*T-i]=p[i]
        elif i==T:
            k=T-i # 0
            for m in range(int(math.floor(T/2)+1)):
                p[i]=p[i]+ ncr(T,2*m) * ncr(2*m, m) * math.pow(p1,m) * math.pow(p2,m) * math.pow((1-p1-p2),(T-2*m))
        else:
            k=T-i #<0
            # if k==1-T:
            #     p[i]= ncr(T,2*m-k) * ncr(2*m, m) * math.pow(p2,-k) * math.pow((1-p1-p2),(T+k))
            # el
            if k==-T:
                p[i]= math.pow(p2,-k)
            else:
                for m in range(int(math.floor(T+k)/2+1)):
                    p[i]=p[i]+ ncr(T,2*m-k) * ncr(2*m-k, m)* math.pow(p1,m) * math.pow(p2,m-k) * math.pow((1-p1-p2),(T-2*m+k))
    return p
p0=custom_random(T,p1,p2)
# print(custom_random(T,p1,p2))
# print(int(np.sum(custom_random(T,p1,p2)))==1)
# print(ncr(T,0)==1)
# print(ncr(0,0)==1)
    

# from scipy.stats import multinomial
# rv = multinomial(8, [0.3, 0.2, 0.5])
# rv.pmf([1, 3, 4])



p1=0.05 # 10^-6, 10^-4
p2=0.03  # 10^-5, 10^-4

#p1=0.003 # 10^-6, 10^-4

lam = 0.005

# z10= z1<p1 # deletion for block
# z11= z1>(1-p2) # insertion for block
# print(z10)


#z20= z2<
from scipy.stats import poisson

#rv = poisson(1000*lam)

def nrandom_poisson(p, lam):
    rv = poisson(lam)
    var, pp = 0, rv.pmf(0)
    while(p>pp):
        var=var+1
        pp=pp+rv.pmf(var)
    return var

def k_map(i,T):
    if i < (2*T+1) and i >= 0:
        k=T-i
        return k
    else:
        print("Input Error!")
        


def nrandom_custom(p, p0):
    var, pp = 0, p0[0]
    while(p>pp):
        var=var+1
        pp=pp+p0[var]
    return k_map(var,T)

# print(nrandom_poisson(0.99,lam*64))

# print(nrandom_custom(0.0002,p0))


L0=60 # initial length of sequence
b0=5 # number of blocks (tandems)
n0=4 # lenght of repeated blocks
m0=0 # number of mutations (SNPs)

gen=np.array((L0,m0))
temp_gen=[gen,gen]
temp_gen=np.repeat(temp_gen,2**(T-1),axis=0)
temp_gen=np.asarray(temp_gen,dtype="float")

z1=np.random.uniform(0,1,len(temp_gen))
z2=np.random.uniform(0,1,temp_gen.shape[0])

p1=np.random.uniform(10**-4,10**-3,len(temp_gen)) # 10^-6, 10^-5
p2=np.random.uniform(10**-3,10**-2,len(temp_gen))  # 10^-5, 10^-4

print(temp_gen.shape)

for i in range(len(temp_gen)):
    temp_gen[i,0]=temp_gen[i,0]+n0*nrandom_custom(z2[i],custom_random(T,p1[i],p2[i]))
    # print(temp_gen[0,0])
    # print(z1[0])
    # print(custom_random(T,p1[0],p2[0]))
    temp_gen[i,1]=nrandom_poisson(z1[i],lam*(L0+ temp_gen[i,0])/2) #emp_gen[0,1]+
    # print(temp_gen[0,1])
    # print(z2[0])

print(custom_random(T,0.0001,0.0001))
print(temp_gen)

#from collections import Counter
import pandas as pd
#count_set=[tuple(element) for element in temp_gen[0,1]]
#ps = pandas.Series([tuple(i) for i in temp_gen[:,1]])
ps = pd.DataFrame(temp_gen)
ps1 = pd.Series(temp_gen[:,0])
counts1 = ps1.value_counts()
print(counts1)
ps2 = pd.Series(temp_gen[:,1])
counts2 = ps2.value_counts()
print(counts2)

#sort_gen = np.array(temp_gen, dtype=np.dtype([('L', int), ('C', int)]))
ind = np.lexsort((temp_gen[:,1],temp_gen[:,0])) # sort by L (0) then by C (1)
print('Sorted indices->',ind)
print(temp_gen[ind,])

#ps3 = pd.Series(temp_gen[:,0]+temp_gen[:,1])
#counts3 = ps3.value_counts()
#print(counts3)

#ps['L'] = ps3
#print(ps.columns)
#ps.columns=['L_orig', 'C','L'] 
#print(ps.columns)
#ps_sort=ps.sort_values(['L','C'], ascending=[True, True])
#print(ps_sort.head)
