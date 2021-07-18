

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages


def dataframe_description(df):
    
    isna = df.isna().sum(axis=0).to_frame()
    isna.columns = ['Total de Missing']
    isna['% de Missing'] = isna['Total de Missing']/df.shape[0]

    dtypes = df.dtypes.to_frame()
    dtypes.columns = ['Tipo']

    descriptions = df.describe(include='all').T    
    
    resume = pd.merge(dtypes, isna, left_index=True, right_index=True)
    resume = resume.merge(descriptions, right_index=True, left_index=True)
        
    return resume

def export_description_continuos_variables(df, target):
    
    pp = PdfPages(r'..\Functions\reports\graficos_variaveis_continuas.pdf')

    for column in df.columns[[x != 'object' for x in df.dtypes]]:
        fig, axis = plt.subplots(1,1, figsize=(20,10))
        x = df[column]
        x.dropna(inplace=True)
        y = df[target]
        y = y.loc[x.index]
        
        try:
            m, b = np.polyfit(x, y, 1)
            axis.plot(x, m*x + b, c='orange', linewidth=3, label = 'y = ({:.2f})x + ({:.2f})'.format(m,b))
            coefficient_of_dermination = r2_score(y,m*x + b)
#             print('Coefficiente of Determination:',coefficient_of_dermination)
#             print('y = ({})x + ({})'.format(b,m))
        except:
            coefficient_of_dermination=np.nan
            print('Regressão Linear não convergiu para ', column)
            
            
        axis.scatter(x,y,s=50, alpha=0.5, label=column)
        axis.set_xlabel(column)
        axis.set_ylabel(target)
        axis.legend(loc='best')
        axis.set_title(column + ' - R²=' + str(round(coefficient_of_dermination,3)))
        plt.show()
        pp.savefig(fig)

    pp.close()



def print_correlation_matrix(df, annot=True):

    # Compute the correlation matrix
    corr = df[df.columns[[x != 'object' for x in df.dtypes]]].corr()

# Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=annot,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
