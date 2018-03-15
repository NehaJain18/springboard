
# Examining Racial Discrimination in the US Job Market

### Background
Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.

### Data
In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.

Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

<div class="span5 alert alert-info">
### Exercises
You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.

Answer the following questions **in this notebook below and submit to your Github account**. 

   1. What test is appropriate for this problem? Does CLT apply?
   2. What are the null and alternate hypotheses?
   3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
   4. Write a story describing the statistical significance in the context or the original problem.
   5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

You can include written notes in notebook cells using Markdown: 
   - In the control panel at the top, choose Cell > Cell Type > Markdown
   - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet


#### Resources
+ Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
+ Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
+ Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
+ Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
</div>
****


```python
import pandas as pd
import numpy as np
from scipy import stats
```


```python
data = pd.io.stata.read_stata('data/us_job_market_discrimination.dta')
```

## Preliminary analysis


```python
# check how the data looks
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ad</th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>...</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
      <th>ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>316</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Nonprofit</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>




```python
# we are interested in race and call columns for now
df = data[['race','call']]

# check data type and if any null values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4870 entries, 0 to 4869
    Data columns (total 2 columns):
    race    4870 non-null object
    call    4870 non-null float32
    dtypes: float32(1), object(1)
    memory usage: 95.1+ KB



```python
# unique values for race
df.race.unique()
```




    array(['w', 'b'], dtype=object)




```python
# unique values for call
df.call.unique()
```




    array([ 0.,  1.])




```python
# number of white-sounding name rows
sum(df['race']=='w')
```




    2435




```python
# number of black-sounding name rows
sum(df['race']=='b')
```




    2435




```python
# number of callbacks for black-sounding names
sum(df[df.race=='b'].call)
```




    157.0




```python
# number of callbacks for white-sounding names
sum(df[df.race=='w'].call)
```




    235.0



## Solutions

The data looks fine and ready for the analysis. Here we are dealing with proportions for two populations i.e. Rate of callbacks for the people with white and black sounding names in their resumes.

We can apply the Z test if the data is coming from normal distribution. CLT will allow us to assume that the 2 sample proportion data is from normal distribution with σ = √[pc*(1-pc)/n1 + pc*(1-pc)/n2] if the following conditions apply:
- The data is collected in random fashion. (Here the same resume was used just with different names and given to different managers.)
- The two samples are independent.
- n1 * p1 >= 5
- n1 * (1-p1) >= 5
- n2 * p2 >= 5
- n2 * (1-p2) >= 5

note: pc is the combined proportion for the 2 samples


```python
# p1 is the callback rate for the white-sounding names
p1 = sum(df.call[df.race=='w'])/sum(df.race=='w')
n1 = sum(df.race=='w')

# p2 is the callback rate for the black-sounding names
p2 = sum(df.call[df.race=='b'])/sum(df.race=='b')
n2 = sum(df.race=='b')

satisfied = (n1*p1 >= 5) & (n1*(1-p1) >= 5) & (n2*p2 >= 5) & (n2*(1-p2) >= 5)

print('CLT conditions satisfied:',  satisfied)
```

    CLT conditions satisfied: True


From above we see that CLT applies in this situation. And hence we can do the Z test. We need to test if both the population proportions are same. Hence our hypothesis is:

        H0: P1 = P2 or P1-P2 = 0
        H1: P1 > P2 or P1-P2 > 0
        
at the significance level of 𝛂 = 0.05

### Testing with two sample proportion Z test


```python
# Let us find the p value

# test statistic here is prop difference
prop_diff = p1-p2
pc = (n1 * p1 + n2 * p2)/ (n1 + n2)
sigma = np.sqrt(pc*(1-pc)/n1 + pc*(1-pc)/n2)

print('prop_diff: ', prop_diff)
print('standard deviation: ', sigma)

# generic formula to get z score:  (test statistic - hypothesized value)/ (stand. deviation of statistic)
z = (p1-p2 -0) / sigma
print('\nz score: ', z)

# p value area of the curve for the specified z value
# (1 - (area of curve above z value) * 2
p = (1 - stats.norm.cdf(z)) * 2
print('p value: ', p)
```

    prop_diff:  0.0320328542094
    standard deviation:  0.00779689403617
    
    z score:  4.10841215243
    p value:  3.98388683758e-05


###### Since p value < 0.05, the data is statistically significant at the significance level of 𝛂 = 0.05 and we reject H0 in favor of H1. It means that there is convincing evidence that there is difference between callback rates for white-sounding vs black-sounding named resumes.


```python
# z_critical at 95%
z_critical = stats.norm.ppf(q=0.975)
print('z_critical: ', z_critical)

# margin of error
moe = z_critical * sigma
print('\nmargin of error: ', moe)

# confidence interval
ci = (prop_diff-moe, prop_diff+moe)
print('95% confidence interval: ', ci)
```

    z_critical:  1.95996398454
    
    margin of error:  0.0152816315022
    95% confidence interval:  (0.016751222707276352, 0.047314485711614819)


### Now let us try Bootstrap method of testing our Hypothesis.




```python
# We can resample from the same population assuming that there is no difference between the two proportions.
def get_prop_diff(sample1, sample2):
    
    p1 = np.sum(sample1['call'] == 1)/len(sample1)
    p2 = np.sum(sample2['call'] == 1)/len(sample2)
    
    return abs(p1-p2)
    
def get_bs_samples_diff(sample1, sample2, func, size):
    length1 = len(sample1)
    length2 = len(sample2)
    bs_prop_diffs = np.empty(size)
    
    for i in range(size):
        combined_sample = pd.concat([sample1,sample2])
        shuffled_sample = combined_sample.sample(length1+length2).reset_index(drop=True)

        new_sample1 = shuffled_sample.iloc[:length1,:]
        new_sample2 = shuffled_sample.iloc[length1:,:]
        
        bs_prop_diffs[i] = func(new_sample1,new_sample2)
        
    return bs_prop_diffs

bs_samples_diff = get_bs_samples_diff(df[df.race=='w'], df[df.race=='b'], get_prop_diff, 10000)
print(bs_samples_diff[:5])
```

    [ 0.00082136  0.00821355  0.00657084  0.00246407  0.01314168]



```python
# p value

p = np.sum(bs_samples_diff > prop_diff)/len(bs_samples_diff)
print('number of times the random selected prop differences is greater than our samples prop diff is: ', p)
```

    number of times the random selected prop differences is greater than our samples prop diff is:  0.0


###### This suggests that there is some impact of white vs black sounding names since the probability of getting as extreme the difference that we see in our samples is 0 in 10000 data points.






### Chi-squared test for nominal (categorical) data

Chi-squared test is used to determine whether an association between 2 two categorical variables in a sample is likely to reflect a real association in the population. The sample data is used to calculate a single number (or test statistic), the size of which reflects the probability (p-value) that the observed association between the 2 variables has occurred by chance, ie due to sampling error.


```python
# Let us see the data in tabular format
contingency_table = pd.crosstab(index=df.race, columns=df.call, margins=True)
contingency_table.index = ['Black', 'White', 'Col_Totals']
contingency_table.columns=['No','Yes', 'Row_Totals']
contingency_table
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>Yes</th>
      <th>Row_Totals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Black</th>
      <td>2278</td>
      <td>157</td>
      <td>2435</td>
    </tr>
    <tr>
      <th>White</th>
      <td>2200</td>
      <td>235</td>
      <td>2435</td>
    </tr>
    <tr>
      <th>Col_Totals</th>
      <td>4478</td>
      <td>392</td>
      <td>4870</td>
    </tr>
  </tbody>
</table>
</div>




```python
observed = contingency_table.ix[0:2,0:2]
observed
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Black</th>
      <td>2278</td>
      <td>157</td>
    </tr>
    <tr>
      <th>White</th>
      <td>2200</td>
      <td>235</td>
    </tr>
  </tbody>
</table>
</div>




```python
expected = np.outer(contingency_table['Row_Totals'][0:2], contingency_table.ix['Col_Totals'][0:2]) / 4870
print(expected)

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
print('chi squared statistic: ', chi_squared_stat)
```

    [[ 2239.   196.]
     [ 2239.   196.]]
    chi squared statistic:  16.87905041427022



```python
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 1)   # *

print("Critical value: ", crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=1)
print("P value: ", p_value)
```

    Critical value:  3.84145882069
    P value:  3.98388683759e-05



```python
# similar results with directly using the stats lib
chi2, p, dof, ex =stats.chi2_contingency(contg_table)
print(ex)
print('degrees of freedom: ', dof)
print('chi squared statistic: ', chi2)
print("P value: ", p)
```

    [[ 2239.   196.]
     [ 2239.   196.]]
    degrees of freedom:  1
    chi squared statistic:  16.4490285842
    P value:  4.99757838996e-05


###### A very high p value indicated that there is some relationship between the variables. And there is impact of black vs white souding names on the callback rates.

### Summary

- We started with checking if our sample was from a normal distribution. We found that given the conditions it was find to assume a normal distribution. Further we could apply the CLT depending on our sample data characteristics.

- To find out if we can determine if there was racial discrimination, we wrote Null Hypothesis that was no discrimination and any difference in the callback rates was due to the chance.

- We did two sample proportion Z and Chi Squared tests. Also tested our hypothesis with bootstrap method.

- All the three tests found the p-value to be very low. That is the probability of actual difference in the 2 proportions was quite significantly low (p-value very near to 0 of order of e-05). We reject the null hypothesis in favor of the alternate hypothesis. There is significant impact of race on the callback rates.

- However our anaylsis does not mean that race is the most important factor. We don't know if those names of the candidates can also be associated somehow with religion, location etc. The problem above says that the resumes are exact match for both the black and white sounding names. So there is no other direct variable other than names that can affect the callback rates. But again the names can be associated with unknown factor.
