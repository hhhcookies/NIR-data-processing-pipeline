#%%
import pandas as pd
import numpy as np

from scipy.signal import savgol_filter
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import KFold
from scipy.stats import f, norm 

import os
from pathlib import Path

import matplotlib.pyplot as plt


################  data and metadata exploration ###############
def data_import(filepath,threshold=4443):
    file = Path(filepath)
    raw_data = pd.read_excel(file,0,header=0,index_col=None)
    columns = list(raw_data.columns)

    metadata_column = [x for x in columns if type(x)==str]
    data_column = [x for x in columns if not type(x)==str]

    selected_columns = [x for x in data_column if x >threshold]

    metadata = raw_data[metadata_column]
    data_nir = raw_data[selected_columns]



    return metadata,data_nir

def metadata_summary(formulation,metadata):
    metadata_info = pd.DataFrame(list(formulation.values()),columns=['Formulation'],index=list(formulation.keys()))
    per_type_count = metadata.groupby(['Type']).count().reset_index()['Label']
    metadata_info['N of samples'] = per_type_count.values

    per_type_operator1 = metadata.groupby(['Type','Operator']).count().reset_index()
    per_type_operator1 = per_type_operator1[per_type_operator1['Operator']=='op1'].reset_index()['Label']
    per_type_operator2 = per_type_count - per_type_operator1

    metadata_info['by operator 1'] = per_type_operator1.values
    metadata_info['by operator 2'] = per_type_operator2.values

    print(metadata_info)
    
    return metadata_info


###################### spectrum preprocessing and plot ################

def SNV(x):
    x_standard = (x-x.mean())/x.std()
    return x_standard


def plot_spectrum(metadata,spectrum,title,wavelength_lb=4000, wavelength_up=7500,threshold = 4443,
                  colorby=None,
                  cmap = plt.cm.viridis):
    
    fig,ax = plt.subplots(1,1,figsize = (12,5))
    spectrum_T = spectrum[spectrum.columns[spectrum.columns>threshold]].T
    
    if colorby:  
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin = metadata[colorby].min(),vmax = metadata[colorby].max())
        color_val = cmap(norm(metadata[colorby]))
        for i in range(spectrum.shape[0]):
            ax.plot(spectrum_T.iloc[:,i],lw=2.5,color = color_val[i])

        sm = plt.cm.ScalarMappable()
        sm.set_array(metadata[colorby])
        cbar = fig.colorbar(sm,ax=ax)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(colorby,fontsize=15)
        
    else:
        ax.plot(spectrum_T,lw=2.5)
    ax.set_xlim(wavelength_lb,wavelength_up)
    ax.invert_xaxis()

    x_labels = ax.get_xticklabels()
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(x_labels,fontsize=15)

    y_labels = ax.get_yticklabels()
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(y_labels,fontsize=15)

    ax.grid(True,which='major',axis='x',color='gray',lw=2,alpha=0.7)

    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=15,fontweight='bold')
    ax.set_title(title,fontsize=15,fontweight='bold')

########################### spectrum pre-processing #######################
def preprocessing_pipeline(metadata,spectrum,smooth_window=31,polyorder=2,SNV=SNV):

    smoothed = savgol_filter(spectrum,smooth_window,polyorder)
    smooth_spectrum_df = pd.DataFrame(data=smoothed,columns=spectrum.columns)
    plot_spectrum(metadata,smooth_spectrum_df,f'smooth with window = {smooth_window}')
    
    if SNV:
        SNVed = smooth_spectrum_df.apply(SNV,axis=1)
        plot_spectrum(metadata,SNVed,f'smooth with window={smooth_window} + SNV')

    processed = SNVed-SNVed.mean()
    plot_spectrum(metadata,processed,f'smooth + SNV + centering')
    return processed


############################## PCA exploratoty analysis ###########################
def pca_hyper(preprocessed_spectrum,n_components=20):

    pca = PCA(n_components=n_components)
    pca.fit(preprocessed_spectrum)

    fig,axes = plt.subplots(2,1,figsize=(10,8),sharex=True)
    axes[0].bar(x=np.arange(1,n_components+1),height=pca.explained_variance_ratio_*100,align='center')
    axes[0].plot(np.arange(1,n_components+1),pca.explained_variance_ratio_*100,lw=3,color='red')
    axes[1].bar(x=np.arange(1,n_components+1),height = np.cumsum(pca.explained_variance_ratio_)*100,align='center')

    axes[0].set_xticks(np.arange(1,n_components+1))
    axes[0].set_xlim([0.4,n_components*0.8])
    xlabel = axes[1].get_xticklabels()
    axes[1].set_xticklabels(xlabel,fontsize=15)
    axes[1].set_xlabel('Principle components',fontsize=15)

    axes[0].set_ylabel("exp var (%)",fontsize=15)
    axes[0].set_yticks(np.array([0,25,50,75,100]))
    ylabel = axes[0].get_yticklabels()
    axes[0].set_yticklabels(ylabel,fontsize=15)
    axes[0].set_ylim([0,100])

    axes[1].set_ylabel("cum exp var (%)",fontsize=15)
    axes[1].set_yticks(np.array([0,25,50,75,100]))
    ylabel = axes[1].get_yticklabels()
    axes[1].set_yticklabels(ylabel,fontsize=15)
    axes[1].set_ylim([0,100])

    plt.subplots_adjust(hspace=0.10)

def Q_T2_outlier_detect(processed_spectrum,n_components,CI_percentile,outlier_threshold):
    # T2--> how far a sample lies from the center of the PCA model, in the space spanned by PCs
    # Q --> how much info is not covered by pca for each sample
    pca = PCA(random_state=42)
    X_transform = pca.fit_transform(processed_spectrum)
    
    loading = pca.components_[0:n_components,:]
    score = X_transform[:,0:n_components]
    eigen = pca.explained_variance_[:n_components]

    n,A = score.shape
    CI_percentile = CI_percentile/100
    T2 = np.sum(score**2/eigen,axis=1)

    Q = np.sum((processed_spectrum.values - score@loading)**2,axis=1)

    # 95% interval for T2 
    
    T2_CI = A*(n-1)/(n-A)*f.ppf(CI_percentile,A,n-A)

    # 95% interval for Q

    eigen_res = pca.explained_variance_[n_components:]
    theta1 = np.sum(eigen_res)
    theta2 = np.sum(eigen_res**2)
    theta3 = np.sum(eigen_res**3)

    h0 = 1-(2*theta1*theta3)/(3*theta2**2)
    z = norm.ppf(CI_percentile)
    
    Q_CI = theta1*( z*(2*theta2)**0.5*h0/theta1 + 1 + theta2*h0*(h0-1)/theta1**2)**(1/h0)

    dist = (T2**2+Q**2)**0.5
    threshold = np.percentile(dist,outlier_threshold)
    dist_df = pd.DataFrame(dist,columns=['distance']).reset_index()

    fig,ax=plt.subplots(1,1,figsize=(12,5))
    ax.scatter(T2,Q)
    ax.scatter(T2[dist>threshold],Q[dist>threshold],color='red',label='possible outlier')
    ax.axhline(y=Q_CI,color='red',ls='--')
    ax.axvline(x=T2_CI,color='red',ls='--')
    ax.legend()

    ax.set_title('Sample plot',fontsize=15)
    ax.set_xlabel("Hotelling T$^{2}$",fontsize=15)
    ax.set_ylabel('Q residuals')
        
    return dist_df,score,loading


def sample_plot(metadata,scores,groupby,categorical,X='PC1',Y='PC3',total_number_of_PC=3):

    pc_map={}
    for i in range(total_number_of_PC):
        pc_map[f'PC{i+1}'] = i
    
    if categorical:
        cat,index = np.unique(metadata[groupby],return_inverse=True)
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin = index.min(),vmax = index.max())
        colorvalue = cmap(norm(index))

        fig,ax = plt.subplots(1,1,figsize=(12,5))
        for i in range(len(cat)):
            ax.scatter(scores[index==i,pc_map[X]],scores[index==i,pc_map[Y]],color=colorvalue[index==i],label = cat[i])
        ax.axhline(y=0,ls='--')
        ax.axvline(x=0,ls='--')
        plt.legend(fontsize=14)
        plt.xlabel(X,fontsize=15)
        plt.ylabel(Y,fontsize=15)
        plt.title('Sample plot',fontsize=15)

    else:
        cmap = plt.cm.binary
        norm = plt.Normalize(vmin = metadata[groupby].min(),vmax = metadata[groupby].max())
        colorvalue = cmap(norm(metadata[groupby]))
        fig,ax = plt.subplots(1,1,figsize=(12,5))
        ax.scatter(scores[:,pc_map[X]],scores[:,pc_map[Y]],color=colorvalue)
        ax.axhline(y=0,ls='--')
        ax.axvline(x=0,ls='--')
        plt.xlabel(X,fontsize=15)
        plt.ylabel(Y,fontsize=15)
        ax.set_title('Sample plot',fontsize=15)
        sm = plt.cm.ScalarMappable(norm,cmap)
        sm.set_array(metadata[groupby])
        cbar = fig.colorbar(sm,ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(groupby,fontsize=14)



def loading_plot(processed_spectrum,loading,selected_PC='PC1',total_number_of_PC=3):

    pc_map={}
    for i in range(total_number_of_PC):
        pc_map[f'PC{i+1}'] = i

    plt.figure(figsize=(12,5))
    plt.plot(processed_spectrum.columns,loading[pc_map[selected_PC],:])
    plt.xlabel(selected_PC,fontsize=15)
    plt.ylabel('loadings on '+selected_PC)
    plt.axhline(y=0,color='black',ls='--')
    plt.gca().invert_xaxis()



############################ Regression analysis ####################################

def optimum_variables(X,y,max_components):
    scores_all = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for component in range(0,max_components):
        pls = PLS(n_components=component+1,scale=False)
        scores = cross_val_score(pls,X,y,cv=cv,scoring='r2')
        scores_all.append(scores.mean())

    plt.figure(figsize=(12,5))
    plt.plot(np.arange(1,max_components+1),scores_all,marker='o',mfc='w')
    plt.xticks(np.arange(1,max_components+1))

    plt.xlabel('latent variables',fontsize=15)
    plt.ylabel('R2 cv', fontsize=15)

    return scores_all


def prediction(pls_model,metadata,processed_spectrum,target,lb = -0.5,ub=3):

    cv = KFold(n_splits=5,shuffle=True)

    predict = cross_val_predict(pls_model,processed_spectrum.values,metadata[target],cv=cv)
    true = metadata[target]
    plt.figure(figsize=(12,5))
    plt.scatter(true,predict,ec='black')
    plt.plot([lb,ub],[lb,ub],ls='--',color='r')
    plt.xlabel('experimental response',fontsize=15)
    plt.ylabel('cross validated response',fontsize=15)

    plt.figure(figsize=(12,5))
    plt.scatter(true,true-predict,ec='black')
    plt.axhline(y=0,ls='--',color='r')
    plt.xlabel('experimental response',fontsize=15)
    plt.ylabel('Residual',fontsize=15)

    







