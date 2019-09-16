import numpy as np
import scipy
import scipy.stats as ss

def calc_2d_pdf(data, pdf_res = 0.3, minval = -10, maxval = 10):
    
    #x_bins = np.linspace(np.amin(data[0,0:]), np.amax(data[0,0:]), pdf_res)
    #y_bins = np.linspace(np.amin(data[1,0:]), np.amax(data[1,0:]), pdf_res)
    x_bins = np.arange(minval,maxval,pdf_res)
    y_bins = np.arange(minval,maxval,pdf_res)
    
    a,b,c = np.histogram2d(data[0,0:],data[1,0:],(x_bins,y_bins), density=True)
    
    return a
    
def integrate_2d(twod_data, minval = -10, maxval = 10, pdf_res = 0.3):
    
    xax_vals = np.arange(minval,maxval,pdf_res)[0:-1] + pdf_res/2
    temp = [np.trapz(y = twod_data[0:,i], x = xax_vals) for i in range(pdf1.shape[1])]
    int_val = np.trapz(y = temp, x = xax_vals)
    return int_val
    
#---------------------------------------------------------------------
#------------------------KL divergence module-------------------------
#---------------------------------------------------------------------

def calc_KL_divergence_2d(dat1, dat2):
    # assuming pdf1 is the 'true' pdf and pdf2 is sampled from it
    
    pdf1 = dat1.copy()#/np.sum(dat1)
    pdf2 = dat2.copy()#/np.sum(dat2)
    pdf1[pdf1 == 0] = 1e-10
    pdf2[pdf2 == 0] = 1e-10
    
    KLD = np.zeros_like(pdf1)
    for i in range(pdf1.shape[0]):
        for j in range(pdf1.shape[1]):
            
            KLD[i,j] = pdf1[i,j] * (np.log10(pdf1[i,j]/pdf2[i,j]))
#             print(pdf1[i,j], pdf2[i,j], (np.log10(pdf1[i,j]) - np.log10(pdf2[i,j])))
    return KLD


#---------------------------------------------------------------------
#------------------------KS test (2d) module--------------------------
#---------------------------------------------------------------------


def fhCounts(x,edge):
    # computes local CDF at a given point considering all possible axis orderings
    
    templist = [np.sum((x[0,0:] >= edge[0]) & (x[1,0:] >= edge[1])), 
                np.sum((x[0,0:] <= edge[0]) & (x[1,0:] >= edge[1])), 
                np.sum((x[0,0:] <= edge[0]) & (x[1,0:] <= edge[1])), 
                np.sum((x[0,0:] >= edge[0]) & (x[1,0:] <= edge[1]))]
    return templist

def kstest_2d(dist1, dist2, alpha = 0.05):
    
    num1 = dist1.shape[1]
    num2 = dist2.shape[1]
    
    KSstat = -np.inf
    
    for iX in (np.arange(0,num1+num2)):
        
        if iX < num1:
            edge = dist1[0:,iX]
        else:
            edge = dist2[0:,iX-num1]
        
#         vfCDF1 = np.sum(fhCounts(dist1, edge)) / num1
#         vfCDF2 = np.sum(fhCounts(dist2, edge)) / num2

        vfCDF1 = np.array(fhCounts(dist1, edge)) / num1
        vfCDF2 = np.array(fhCounts(dist2, edge)) / num2
        
        vfThisKSTS = np.abs(vfCDF1 - vfCDF2)
        fKSTS = np.amax(vfThisKSTS)
        
        if (fKSTS > KSstat):
            KSstat = fKSTS
            #print(KSstat, vfCDF1, vfCDF2)

    # Peacock Z calculation and P estimation

    n =  num1 * num2 /(num1 + num2)
    Zn = np.sqrt(n) * KSstat;
    Zinf = Zn / (1 - 0.53 * n**(-0.9));
    pValue = 2 *np.exp(-2 * (Zinf - 0.5)**2);

    # Clip invalid values for P
    if pValue > 1.0:
        pValue = 1.0
        
#     H = (pValue <= alpha)
    
    return pValue, KSstat

#---------------------------------------------------------------------
#------------------------Hotelling T2 test module---------------------
#---------------------------------------------------------------------


def hotelling_T2(dist1, dist2):
    
    # Hotelling's T-Squared test for comparing d-dimensional data from two independent samples, 
    # assuming normality w/ common covariance matrix
    
    if dist1.shape[0] != dist2.shape[0]:
        print('Error: ordered pair dimensions do not match')
    else: 
        p = dist1.shape[0]
        
    n = dist1.shape[1]
    m = dist2.shape[1]
    
    n = n+m
    mux = np.mean(dist1,1)
    muy = np.mean(dist2,1)
    
    Sx = np.cov(dist1)
    Sy = np.cov(dist2)
    
    Su = (n*Sx + m*Sy) / (n-2) # unbiased estimate
    d = np.zeros((p,1))
    d[0:,0] = mux - muy
    
    D2 = np.matmul(d.T,np.matmul(np.linalg.inv(Su),d)) # reversing the formula for python's matrix handling
    #D2 = d.T*np.linalg.inv(Su)*d
    D2 = D2[0][0]
    T2 = ((n*m)/n)*D2
    F = T2 * (n-p-1) / ((n-2)*p);

    pval = 1 - scipy.stats.f.cdf(F,p,n-p-1);
    return pval, T2

#---------------------------------------------------------------------
#------------------------Hotelling T2 test module---------------------
#---------------------------------------------------------------------

def energy_statistics_test(dist1, dist2, alpha = 0.05, flag = 'szekely-rizzo', nboot = 100, replace = False, progbar = False):
    
    if dist1.shape[0] != dist2.shape[0]:
        print('Error: ordered pair dimensions do not match')
        
    n = dist1.shape[1]
    m = dist2.shape[1]
    
    pooled = np.hstack((dist1,dist2))
    
    e_n = energy(dist1, dist2, flag)
    e_n_boot = np.zeros((nboot, ))
    e_n_boot[0] = e_n
    
    if progbar == True:
        
        for i in tqdm(range(1,nboot)):
            
            if replace == False:
                ind = np.random.choice(n+m, size=(n+m,))
            else:
                ind = np.random.permutation(np.arange(n+m))
            e_n_boot[i] = energy(pooled[0:,ind[0:n]],pooled[0:,ind[n:]], flag)
    else: 
        for i in (range(1,nboot)):
            
            if replace == False:
                ind = np.random.choice(n+m, size=(n+m,))
            else:
                ind = np.random.permutation(np.arange(n+m))
            e_n_boot[i] = energy(pooled[0:,ind[0:n]],pooled[0:,ind[n:]], flag)
    
    p = np.sum(e_n_boot>=e_n)/nboot
    
    return p, e_n, e_n_boot
    

def dist(x,y):
    dx = scipy.spatial.distance.pdist(x,'euclidean')
    dy = scipy.spatial.distance.pdist(y,'euclidean')
    dxy = scipy.spatial.distance.cdist(x.T,y.T,'euclidean')
    return dx,dy,dxy

def energy(x,y,flag):
    n = x.shape[1]
    m = y.shape[1]
    dx, dy, dxy = dist(x,y)
    
    if flag == 'aslan-zech':
        temp = -np.log(dxy)
        z = (1/(n*(n-1)))*np.sum(-np.log(dx)) + (1/(m*(m-1)))*np.sum(-np.log(dy)) - (1/(n*m))*np.sum(temp[(temp>-np.inf) & (temp<np.inf)])
    elif flag == 'szekely-rizzo':
        z = (2/(n*m))*np.sum(dxy) - (1/(n**2))*np.sum(2*dx) - (1/(m**2))*np.sum(2*dy)
        z = ((n*m)/(n+m))*z
    else:
        z = np.nan
        print('did not understand method input')
        
    return z
    
    
    
