#this script should be run for both males and females

import numpy as np
from scipy.stats import t
import pandas as pd
import scipy.stats

#to calculate facial expressions we will use Azure's FACE API https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236

def welch_ttest(x1, x2,alternative):

    n1 = x1.size
    n2 = x2.size
    m1 = np.mean(x1)
    m2 = np.mean(x2)

    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)

    pooled_se = np.sqrt(v1 / n1 + v2 / n2)
    delta = m1-m2

    tstat = delta /  pooled_se
    df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))

    # two side t-test
    p = 2 * t.cdf(-abs(tstat), df)

    # upper and lower bounds
    lb = delta - t.ppf(0.975,df)*pooled_se
    ub = delta + t.ppf(0.975,df)*pooled_se

    return pd.DataFrame(np.array([tstat,df,p,delta,lb,ub]).reshape(1,-1),
                        columns=['T statistic','df','P-value 2-sided','Difference in mean','Lower bound','Upper bound'])

def calculate_spearman_ci(r,n):
    try:
        z = (1/2)*math.log((1+r)/(1-r))
        g = z+(1.96)/math.sqrt(n-3)
        f = z-(1.96)/math.sqrt(n-3)
        lower = (math.exp(2*f)-1)/(math.exp(2*f)+1)
        upper = (math.exp(2*g)-1)/(math.exp(2*g)+1)

        return(np.round([lower,upper],3))
    except:
        pass

#read files

files_with_candidates = pd.read_csv("files_with_candidates.csv") #possibly merge data on emotions if not already present
elected = files_with_candidates[files_with_candidates["class"]=="elected"]
not_elected = files_with_candidates[files_with_candidates["class"]=="not_elected"]

#calculate p-values

list_of_p_values = []


#possibly change name of emotions if they have changed in the API
for hest in ['emotion_anger   ','emotion_contempt    ', 'emotion_disgust ', 'emotion_fear    ','emotion_happiness   ',
             'emotion_neutral ', 'emotion_sadness ','emotion_surprise    ']:
    temp = welch_ttest(elected[hest],not_elected[hest], alternative = "unequal")
    list_of_p_values.append(temp)

mean_dif = pd.concat(list_of_p_values)

mean_dif[['P-value 2-sided', 'Difference in mean',
          'Lower bound', 'Upper bound']] = mean_dif[['P-value 2-sided', 'Difference in mean',
                                                     'Lower bound', 'Upper bound']].round(3)
mean_dif["Expression"] = ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
mean_dif = mean_dif[['Expression','T statistic', 'df', 'P-value 2-sided', 'Difference in mean',
                     'Lower bound', 'Upper bound']]
mean_dif.to_latex("expressions_mean_dif.tex", index=False)


#calculate besd and spearman correlations

#read file with probability of getting elected for each candidate
files_overview_candidates = read_csv("files_overview_candidates.csv")


list_of_p_values_cor = []
list_of_pearson_cor = []

for hest in ["probability",'emotion_anger   ','emotion_contempt    ', 'emotion_disgust ', 'emotion_fear    ','emotion_happiness   ', 'emotion_neutral ', 'emotion_sadness ','emotion_surprise    ']:
    temp = scipy.stats.spearmanr(files_overview_candidates['probability'],files_overview_candidates[hest])
    #print(temp)
    list_of_p_values_cor.append(temp[1])

    temp = scipy.stats.pearsonr(files_overview_candidates['probability'],files_overview_candidates[hest])
    list_of_pearson_cor.append(temp[0])



files_overview_candidates['probability'] = files_overview_candidates['probability'].astype("float")
cor_stuff_1 = files_overview_candidates[['probability','emotion_anger   ',
                                         'emotion_contempt    ', 'emotion_disgust ', 'emotion_fear    ',
                                         'emotion_happiness   ', 'emotion_neutral ', 'emotion_sadness ',
                                         'emotion_surprise    ']]
cor_stuff = cor_stuff_1
cor_stuff.columns = ["Probability", "Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
cor_stuff = cor_stuff.corr(method="spearman")

#create dataframe
cor_stuff = pd.DataFrame(cor_stuff.round(3).iloc[:,0])
cor_stuff.columns = ["Correlation"]
cor_stuff["p-value"] = list_of_p_values_cor
cor_stuff["p-value"] = cor_stuff["p-value"].round(3)
cor_stuff["Confidence interval"] = cor_stuff["Correlation"].apply(lambda x: calculate_spearman_ci(x,files_overview_candidates.shape[0]))

#add BESD
cor_stuff["BESD"] = list_of_pearson_cor
cor_stuff["BESD"] = cor_stuff["BESD"].apply(lambda x: 0.5+x/2).round(3)
cor_stuff.to_latex("cor_expressions.tex") #do for both males and females
