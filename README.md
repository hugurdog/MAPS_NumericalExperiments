# MAPS_NumericalExperiments
This repository contains the Numerical Experiments performed to justify the use of the so called MAPS estimators to direct the eigen vector shrinkage of the covariance matrices. This experiments are reported in my Ph.d. thesis at Florida State University, Mathematics Department. 

The following is the current ingredients and their description of the repository, 

(1) MAPSWithSimulatedBetas.py:
This performs the experiments that are reported in the section 3.1 of the dissertation titled: "EIGENVECTOR SHRINKAGE FOR ESTIMATING COVARIANCE MATRICES"

(2) SingleBlock.py: 
This performs the experiments that are reported in the sub-section 3.2.1 of the dissertation titled: "EIGENVECTOR SHRINKAGE FOR ESTIMATING COVARIANCE MATRICES"

(3) DynamicalMAPSWithHistoricalBetas.py:
This performs the experiments that are reported in the sub-section 3.2.2 of the dissertation titled: "EIGENVECTOR SHRINKAGE FOR ESTIMATING COVARIANCE MATRICES"

(4) WRDSBETASNEW.csv: 
This document is referred by the items (2) and (3) above. This document originally contains the historical betas that is obtained via the WRDS beta's for a 24 months period. However, except the row names(tickers) and the column names(months) it is deleted due to the copy-right issues. In order to re-produce the code, the user has to obtain the betas from the WRDS using the beta's module and the following settings,

(a) Elect the time period as 01/19 - 12/20 that amounts to 24 months.
(b) Use the tickers list in the file WRDSBETASNEW.csv without changing the order.  The tickers are grouped according to the sector separation determined by the Global Industr Classification Standard.
(c) Elect the model as "Market Model". This generates the "CAPM" betas. 
(d) Elect the frequency as months, time window as 12 months and the minimum time window as 12 months.

 
