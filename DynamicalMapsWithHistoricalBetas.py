# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 10:32:05 2021

@author: Hubeyb Gurdogan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState as srng
import os
import seaborn as sns
import pandas as pd
from scipy.linalg import eigh

def betas(mu_beta, sig_beta, fmrng, ro, model,*args):
    
    '''
    
    Creates beta1 and beta2 that are standing for the betas of the first and second block of the
    return data. It can do this in three ways that can be set by the user using the "model" parameter. 
    
    The possible values for the model parameter are data2" and "sim":
     % When model is set to "data2", the function draw beta1 and beta2 from data which possibly represent
    the betas of two subsequent data blocks.
    % When model is set to "sim" the function generate beta1 and beta2 randomly from
    a normal distribution with mean and standard deviation set to mu_beta and sig_beta respectively.
    Also, the function introduces a correlation between beta1 and beta2 that is set by the user via the 
    variable "ro". If the user sets "ro" to 1, that will force beta1 and beta2 equal to each other. 

    Parameter List:
    
    mu_beta  : mean of the beta's, if it is created randomly via setting model parameter to "sim"
    sig_beta : standard deviation of the beta's if it is created randomly via setting model parameter to "sim"
    fmrng    : random number generator that is an object of the numpy.random.RandomState class.
    ro       : correlation of beta1 and beta2, if it is created randomly via setting model parameter to "sim"
    model    : Can take values "data1", "data2" and "sim", explained above. 
    *args    : it is a list of two parameters that encodes the time window numbers we are getting our betas from.
              
    Example:
        fmrng=srng(1)
        beta1, beta2= betas(1, .33, fmrng, .7, "data2", '5', '6' )
    
    
    '''
     
    #  Below we read from the *args which holds time window numbers we are interested in.
    m=0
    for arg in args:
        m=m+1
    
    s=np.zeros((m,1))
    i=0
    
    for arg in args:
        s[i]=arg
        i=i+1
        
    
    # We draw the betas for double block data, from the csv file obtained from S&P 500 data.
   
   
    if model=='data2':
        
        
        
        df1=pd.read_csv("C:/Users/16692/Desktop/NewData/WRDSBetasNew.csv")
        beta1=np.zeros((p,1))   # This will be the beta governing the first data block
        beta2=np.zeros((p,1))   # This will be the beta governing the second data block
        for i in range(p):
            beta1[i]=df1.iloc[i][int(s[0])]
            beta2[i]=df1.iloc[i][int(s[1])]
        
        
    # Below will create a randomly generated double or single block depending on the rho selection.    
    # if rho=1, it will correspond to the single block of data.    
    
    if model=='sim':
        
        raw1 = fmrng.normal(0,1,p).reshape((p,1))
        raw2 = fmrng.normal(0,1,p).reshape((p,1))
        temp = ro*raw1+np.sqrt(1-ro*ro)*raw2   
        beta1 = mu_beta+sig_beta*temp
        beta2 = mu_beta+sig_beta*raw1
        
        
    
    return (beta1, beta2)

    
    
    
def MgenerateY(n1,n2,sig_z,mu_beta,sig_beta,p,fmrng,ro, model,*args):
    
    '''
    
    This function creates double block of return data for three different model
    specifications that are detailed below. 
    
    ## For "data1", it draws the beta1 from the data and sets beta2=beta1 using the
    betas function. Then it creates two blocks of return data using these betas and a double block
    by concatenating them. Since beta1=beta2 it essentially creates a single block of size n1+n2 for
    our purposes. 
    
    ## For "data2" it does follow the same routine to draw the beta1 and beta2 from the data and creates
    a double block of data where each block within is created around beta1 and beta2. 
    
    ## For "sim" it creates a double blokc of return data randomly where blocks use beta1 and beta2
    that are correlated by "ro". If "ro" is set to 1, that will set beta1=beta2, resulting into a 
    single block of data for our purposes. 
    
    '''
    
    
    X=fmrng.normal(0,sig_x,n1+n2).reshape((n1+n2,1))
    X1=X[0:n1,:]
    X2=X[n1:n1+n2,:]
   
    
    beta1, beta2= betas(mu_beta, sig_beta, fmrng, ro, model,*args) # Note here that we may need to use a mean vector instead of a constant mean  
    Z=fmrng.normal(0,sig_z,p*(n1+n2)).reshape((p,n1+n2))
    Z1=Z[:,0:n1]
    Z2=Z[:,n1:n1+n2]
    
    R1=np.dot(beta1,X1.transpose())+Z1
    R2=np.dot(beta2,X2.transpose())+Z2
    
    R=np.concatenate((R1,R2),axis=1)
    
    return(beta1,beta2,R1,R2,R) 

def necessaryEigenStruct(Y,p,n):
    
    '''
    
    Returns the leading eigenvector of the sample covariance matrix of Y  and also
    it returns the variable "phi_square" that is the estimation of the inner product between the 
    leading eigenvector and the normalized true beta. 
    
    '''

    Sigma_hat = np.dot(Y,Y.transpose())/n
    
    eigvalue,beta_hat= eigh(Sigma_hat, eigvals=(p-1,p-1))
    beta_hat = np.real(beta_hat[:,0]).reshape((p,1)) 
    
    trace=np.trace(Sigma_hat)
    
    phi_square=eigvalue-((trace-eigvalue)/(n-1)) # this follows the formulation in minvar paper.
    phi_square=phi_square/eigvalue  
    
    if beta_hat.sum()<0:
        beta_hat = -beta_hat

    return {'h':beta_hat,'phi_square': phi_square}
      
##############################################
######### Anchor points selection ############
#############################################
    

def seperate_rule(h,k,p):
    '''
        return k * p matrix, each row of which represents one normalized anchor point
    '''
    start = 0
    step = int(np.floor(p/k))
    fill = np.sqrt(1/step)
    result = np.zeros((k,p))
    sort_ = h[:,0].argsort()
    
    if k>1:
        for i in range(k-1):
            result[i,sort_[start:(start+step-1)]] = fill
            start += step
        
        result[i+1,sort_[start:]] = 1/np.sqrt(p-start)
    else:
        result = result+1
           
    return result

def seperate_rule_v2(k,fmrng,p):
    '''
        return k * p matrix each row of which represents one normalized anchor point
    '''
    uniform_prob = fmrng.uniform(0,1,p)
    prob = 1/k
    lb = 0
    ub = lb+prob
    
    result = np.zeros((k,p))
    
    if k>1:
        for i in range(k-1):
            temp = np.where(np.logical_and(uniform_prob>=lb,uniform_prob<ub))[0]
            result[i,temp] = np.sqrt(1/temp.shape[0])
            lb = lb+prob
            ub = lb+prob
            
        temp = np.where(uniform_prob>=lb)[0]
        result[i+1,temp] = np.sqrt(1/temp.shape[0])

    else:
        result = result+1/np.sqrt(p)
           
    return result




def seperate_rule_sector(p,state):
    
    '''
    
    This is the routine to generate partition that is not neccesarily the beta-ordered. 
    The csv files bss1-bss6 contains the PCA betas for the S&P 500 stocks ordered by the industries.
    Below we implement a sector seperation. 
    
    '''
    
    if state=='calm':
        ss=[255,240] # InfoTech	Energy	Industrials	Financials	Materials ----- Consumer D	Communication SRVC	Health Care	Consumer S	Real Estate	Utilitites
        k=2
    
    elif state=='stressed':
      
        k=11
        ss=[71,23,66,64,28,63,25,58,32,30,28] # this contains the number of tickers in each sector
        
   
    start = 0
    result = np.zeros((k,p))
    for i in range(k):
        result[i,start:(start+ss[i])] =np.sqrt(1/ss[i]) 
        start += ss[i]
        
    
    return result

def seperate_rule_cap(p):
    
    '''
    
    This is the routine to generate partition that is not neccesarily the beta-ordered. 
    The csv files bss1-bss6 contains the PCA betas for the S&P 500 stocks ordered by the industries.
    Below we implement a sector seperation. 
    
    '''
    #k=5
    k=2
    
    df=pd.read_csv(r'C:\Users\16692\Desktop\NewData\CAP.csv')
    result = np.zeros((k,p))
    for i in range(p):
        if df.iloc[i][1]<2:
            result[0][i]=1
        else:
            result[1][i]=1
    e=np.ones((p,1))
    ad=np.dot(result,e)

    for j in range(k):
        result[j,:]=result[j,:]/np.sqrt(ad[j])
        
    
    return result


#########################################################
################ Error Metrics ##########################
#########################################################


def vecNorm(b):
    return (b*b).sum()


def diffNorm(a,b):
    return ((a-b)*(a-b)).sum()

def OptBias(h,b,p):
    
    q = np.ones((p,1))
    q = q/np.sqrt(vecNorm(q))
    
    bq = np.dot(b.transpose(),q)
    hq = np.dot(h.transpose(),q)
    hb = np.dot(b.transpose(),h)
    
    E = bq - hq*hb
    E = E/(1-hq*hq)
    
    return E*E
    
def tracking_error(K_actual, K_estimated, v, b, beta, q, sig_x, sig_z):
    
    
    bq=np.dot(b.transpose(),q)
    vq=np.dot(v.transpose(),q)
    
    r_actual=(1+K_actual)/bq
    r_estimated=(1+K_estimated)/vq
    
    w_actual=r_actual*q-b
    w_actual=w_actual*(1/(np.sqrt(p)*(r_actual-bq)))
    
    w_estimated=r_estimated*q-v
    w_estimated=w_estimated*(1/(np.sqrt(p)*(r_estimated-vq)))
    
    temp=np.dot(w_estimated.transpose()-w_actual.transpose(),beta)
    tre=sig_x*sig_x*temp*temp+sig_z*sig_z*vecNorm(w_estimated-w_actual)
    tre=tre[0]
    tre=tre[0]
    tre=np.abs(tre)
    
    return(tre)

########################################
################ Main Method ##########
#######################################
    

def orthogonalize(L):
    '''
    L: pxk matrix that holds the basis vectors of the subspace on its columns. 
    
    This function takes a matrix of size pxk and orthonormalize its coulumns. It
    also replaces a vector by 0 if it is near linearly dependent with the previous vectors
    in the gram schmidt process.
    '''
    
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    
    eps=0.000000000000001
    (p,k)=L.shape
    if k<2:
        return L
    U = L.transpose()
    for i in range(k):
        prev_basis = U[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, U[i].transpose())  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        U[i] -= np.dot(coeff_vec, prev_basis).transpose()
        if np.linalg.norm(U[i]) < eps:
            U[i][U[i] < eps] = 0.   # set the small entries to 0
        else:
            U[i] /= np.linalg.norm(U[i])
    return U.transpose()

def project(v,L):
    '''
    v: given vector
    L: pxk matrix that holds the basis vectors of the subspace on its columns. 
    
    This function first orthonormalize the given pxk matrix and then projects the given vector onto
    the linear subspace spanned by the columns of the orthonormalized matrix.
    '''
    # note that v is a px1 vector.
    (p,k)=L.shape
    L=orthogonalize(L)
    U=L.transpose()
    u=np.zeros((1,p))
    for i in range(k):
        u=u+np.dot(U[i],v)*U[i] 
    
    return u.transpose() # now the output is px1 vector too. 

def GPS(p,n,eigen_result,h):
    beta_hat = h
    phi_square = eigen_result['phi_square']
    #print(phi_square)
    rho = beta_hat.sum()/np.sqrt(p)*(1-phi_square)/(phi_square-(beta_hat.sum()/np.sqrt(p))**2)
    beta_hat = beta_hat +rho/np.sqrt(p)
    beta_hat = beta_hat/np.sqrt((beta_hat*beta_hat).sum())

    return beta_hat

        
def multi_anchor_points_shrinkage(p,n,k0,eigen_result,h1,b,method,*argvs):
    h2 = eigen_result['h']  # h2 will be used to get an estimation of b
                           
    phi_square=eigen_result['phi_square']
    
    if method == "Dynamical_MAPS with Sector Seperation":
        
        q=np.ones((p,1))
        fill=1/np.sqrt(p)
        q=fill*q
        
        
        anchor_points2= np.zeros((12,p))    
        anchor_points2[0,:]=h1.transpose()
    
        for i in range(11):
            anchor_points2[i+1,:]=q.transpose()
        
        
        
        
    if method == "Dynamical_MAPS":
        

        q=np.ones((p,1))
        fill=1/np.sqrt(p)
        q=fill*q
        
        anchor_points = np.zeros((2,p))
        anchor_points[0,:]=h1.transpose()
        anchor_points[1,:]=q.transpose()
      
        
    elif method == "random":
        if not 'fmrng' in argvs[0].keys():
            raise Exception('no fmrng input')
        anchor_points =seperate_rule_v2(k0,argvs[0]['fmrng'],p)
             
    elif method == 'beta_ordered':
        
        anchor_points = seperate_rule(b,k0,p) 
    
    elif method == 'sector_calm':
        
        anchor_points = seperate_rule_sector(p,'calm')
        
    elif method == 'sector_stressed':
        
        anchor_points = seperate_rule_sector(p, 'stressed')
    
    elif method == 'cap':
        
        anchor_points = seperate_rule_cap(p)
                        
                    
    else:
        raise Exception('wrong method')
     
    q=np.ones((p,1))
    fill=1/np.sqrt(p)
    q=fill*q
    
    L=anchor_points.transpose()
    L=orthogonalize(L) # not necessary if we are using the seperation based subspaces.
    v=project(h2,L)
    if (q*h2).sum()<0:
        print(method)
    tau=(phi_square-vecNorm(v))/(1-phi_square)
    hd=tau*h2+v
    hd=hd/np.sqrt(vecNorm(hd))

    return hd

########################################################
############ SIMULATION ################################
########################################################    

def simulation(seed,p,n,k,sig_x,sig_z,mu_beta,sig_beta,ro,s1,s2):
    
    fmrng=srng(seed)
    n1=int(n/2)
    n2=int(n/2)
    
    
    '''
    
    Creating the model as a two block model where depending on the choice, the two blocks can share the 
    same beta, making it a single block data for our purposes. Please see "MgenerateY" and "betas" for
    detailed information.
    
    '''
    """
    x=np.arrange(0,1,.01)
    for i in range(100):
        beta1,beta2,Y1,Y2,Y= MgenerateY(n1,n2,sig_z,mu_beta,sig_beta,p,fmrng,x[i], 'sim','5','6')
                
    """
    
    
    q=np.ones((p,1))
    fill=1/np.sqrt(p)
    q=fill*q
    
    beta1,beta2,Y1,Y2,Y= MgenerateY(n1,n2,sig_z,mu_beta,sig_beta,p,fmrng,ro,'data2',s1,s2)
    b= beta2/np.sqrt((beta2*beta2).sum()) # We are after the final beta
    eigen_result3 = necessaryEigenStruct(Y,p,n) # Merged data block
    eigen_result2 = necessaryEigenStruct(Y2,p,n2) # Current data block
    eigen_result1 = necessaryEigenStruct(Y1,p,n1) # Past data blockss
    
    h3= eigen_result3['h'] #PCA of the double block
    h2= eigen_result2['h'] #PCA of the current block 
    h1= eigen_result1['h'] #PCA of the past block
    
    
    
    hg2=GPS(p,n,eigen_result3,h3) # double block gps
    hg1=GPS(p,n1,eigen_result2,h2) # single(current) block gps
    ho2=multi_anchor_points_shrinkage(p,n,k,eigen_result3,b,b,'beta_ordered') 
    hd =multi_anchor_points_shrinkage(p,n1,k,eigen_result2,h1,b, 'Dynamical_MAPS') 
    result1 = {}
    result1['k'] = k
    result1['p'] = p
 
    
    ###########################################
    ####### Tracking error calculation ########
    ###########################################
    
    q=np.ones((p,1))
    fill=1/np.sqrt(p)
    q=fill*q

    phi_squared2=eigen_result2['phi_square']
    K_actual=(sig_z*sig_z)/(sig_x*sig_x*vecNorm(beta2))
    K_estimated3=(n/p)*(1/phi_squared2-1)
    
    
    To2=tracking_error(K_actual, K_estimated3, ho2, b, beta2, q, sig_x, sig_z)
    Tg2=tracking_error(K_actual, K_estimated3, hg2, b, beta2, q, sig_x, sig_z)
    Tg1=tracking_error(K_actual, K_estimated3, hg1, b, beta2, q, sig_x, sig_z)
    Td=tracking_error(K_actual, K_estimated3, hd, b, beta2, q, sig_x, sig_z) # Picking K_estimated3 created a better portfolio in an odd way.
    Tp1=tracking_error(K_actual, K_estimated3, h2, b, beta2, q, sig_x, sig_z)
    Tp2=tracking_error(K_actual, K_estimated3, h3, b, beta2, q, sig_x, sig_z)
    
    
    result1['PCA1']=p*Tp1
    result1['PCA2']=p*Tp2
    result1['GPS1']=p*Tg1
    result1['Dynamical MAPS']=p*Td
    result1['GPS2']=p*Tg2
    result1['Beta Ordered']=p*To2 
    
    
    ############################################
    ############ l2 Norm Calculation ###########
    ############################################
    result2 = {}
    result2['k'] = k
    result2['p'] = p
    result2['PCA1']=diffNorm(h2,b)
    result2['PCA2']= diffNorm(h3,b)
    result2['GPS1']=diffNorm(hg1,b)
    result2['Dynamical MAPS']=diffNorm(hd,b)
    result2['GPS2']=diffNorm(hg2,b)
    result2['Beta Ordered']=diffNorm(ho2,b)
    
    
    
    
    ######################################################
    ############ Optimization Bias Calculation ###########
    ######################################################
    result3 = {}
    result3['k'] = k
    result3['p'] = p
    result3['PCA1']=p*OptBias(h2,b,p)
    result3['PCA2']=p*OptBias(h3,b,p)
    result3['GPS1']=p*OptBias(hg1,b,p)
    result3['Dynamical MAPS']=p*OptBias(hd,b,p)
    result3['GPS2']=p*OptBias(hg2,b,p)
    result3['Beta Ordered']=p*OptBias(ho2,b,p)
    
    
    
    return result1, result2, result3 #, dd


def plot_box(df,save_path,name, xname, yname,upper_bound):
    # init
    df = rearrange_df(df,xname,yname)
    g = (1 + np.sqrt(5))/2
    f,ax = plt.subplots( figsize=(8,8/g))
    
    # check file
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # box
    if "original" in name.lower():
        color_name = 'Set1'
    else:
        color_name = 'Set3'
    sns.boxplot(x=xname, y=yname, hue="method",data=df, palette=color_name,ax = ax)
    ax.set(ylim=(0,upper_bound))
    
    # legend
    plt.legend(loc='best')
    
    # title
    
    plt.title(name,fontsize = 15)
    
    # save
    f.savefig("{0}{1}.png".format(save_path,name.replace(" ","_")),dpi = 400)
    return


def rearrange_df(df,xname, yname):
    exclude_col = ['p','k','multi_GPS']
    result = []
    for col in df.columns:
        if col in exclude_col:
            continue
        temp = df.loc[:,['p',col]]
        temp.columns = [xname,yname]
        temp['method'] = col
        result.append(temp)
    
    return pd.concat(result)

if __name__ == "__main__":
    # init
    seed_list = range(100)
    p_start =488
    p_step =0
    p_end =488
    n = 24
    sig_x =.16
    sig_z = .5
    mu_beta =1
    sig_beta =.5
   
    p=p_start
    El2errors_list = []
    ETracking_errors_list = []
    Opt_Bias_list = []
    for j in range(1,13):
        El2errors = {}
        ETracking_errors = {}
        Opt_Bias = {}
        result1 = []
        result2 = []
        result3 = []
        for seed in seed_list:
                print(j,seed)
                r1,r2,r3=simulation(seed,p,n,6,sig_x,sig_z,mu_beta,sig_beta,(j/10),j,j+12)
                result1.append(r1)
                result2.append(r2)
                result3.append(r3)
                
                result_df1 = pd.DataFrame(result1)
                result_df2 = pd.DataFrame(result2)
                result_df3 = pd.DataFrame(result3)
        
        for col in result_df1:
            ETracking_errors[col] = result_df1[col].mean()
        for col in result_df2:
            El2errors[col] = result_df2[col].mean()
        for col in result_df3:
            Opt_Bias[col] = result_df3[col].mean()
        
        ETracking_errors_list.append(ETracking_errors)
        El2errors_list.append(El2errors)
        Opt_Bias_list.append(Opt_Bias)
    
    El2_df = pd.DataFrame(El2errors_list)
    El2_df=El2_df.drop(['k','p'], axis=1)
    
    ETracking_df = pd.DataFrame(ETracking_errors_list)
    ETracking_df = ETracking_df.drop(['k','p'],axis=1)
    Opt_Bias_df = pd.DataFrame(Opt_Bias_list)
    Opt_Bias_df = Opt_Bias_df.drop(['k','p'], axis=1)
    #plot_box(El2_df,"C:/Users/16692/Desktop/NewData/img/Empirical Single Block Mean L_2 Error","","","L_2 Error",1)
   
    g = (1 + np.sqrt(5))/2
    f,ax = plt.subplots( figsize=(8,8/g))
    sns.boxplot(data=El2_df, ax=ax) 
    ax.set(ylim=(0,1))
    ax.set(xlabel="Method", ylabel = "Expected L_2 Error")
    plt.savefig('DoubleleBlock_L_2')
    plt.close()
    
    g = (1 + np.sqrt(5))/2
    f,ax = plt.subplots( figsize=(8,8/g))
    sns.boxplot(data=ETracking_df, ax=ax) 
    ax.set(ylim=(0,8))
    ax.set(xlabel="Method", ylabel = "Expected Tracking Error x p")
    plt.savefig('DoubleBlock_Tracking')
    plt.close()
    
    g = (1 + np.sqrt(5))/2
    f,ax = plt.subplots( figsize=(8,8/g))
    sns.boxplot(data=Opt_Bias_df, ax=ax) 
    ax.set(ylim=(0,200))
    ax.set(xlabel="Method", ylabel = "Expected Optimization Bias x p")
    plt.savefig('DoubleBlock_Opt_Bias')
    plt.close()
    

    