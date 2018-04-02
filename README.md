# DistanceLearning
An R package that implements 11 distance metric learning methods. The implemented methods are: 
<br/>
<br/>
1- Free Energy Nearest Neighbor (FENN) <br/>
2- Free Energy Nearest Neighbor with dimension reduction (subFENN)<br/>
3- Discriminant Adaptive Nearest Neighbor (DANN)<br/>
4- iterative Discriminant Adaptive Nearest Neighbor (iDANN)<br/>
5- Locally Adaptive Metric Nearest Neighbor (ADAMENN)<br/>
6- Locally Adaptive Metric Nearest Neighbor (iADAMENN)<br/>
7- Xing's method<br/>
8- Relevant Component Analaysis (RCA)<br/>
9- Linear Fisher Discriminant Analaysis (LFDA)<br/>
10- Discriminant Component Analaysis (DCA)<br/>
11- Neighborhood Component Analysis (NCA)<br/>

All methods are implemented using `RcppArmadillo` in order to attain improved performance. A vignette detailing the usage of each method is attached to the package.


## Installation
You can install this R pacakge using the following:
```{R}
library(devtools)
install_github("carltonyfakhry/DistanceLearning")
```
## Vignette
Please see the *Vignette* for this package using the following:
```{R}
browseVignettes("DistanceLearning")
```
