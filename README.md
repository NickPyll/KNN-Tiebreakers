# KNN-Tiebreakers
Investigating how R breaks ties in KNN Classification

---
author: "Nicholas Pylypiw"

date: "July 6, 2017"

---

One of the potential problems that comes up when using K-nearest neighbor classification methods is what to do in the event of a tie.  One of the common 'remedies' I have heard for this issue is to simply choose an odd number for `k`.  Apart from my philosophical problems with arbitrarily deviating from the optimal `k` value, this practice does not always work.  For example, when using the Iris dataset, which has three species, a tie is still possible when using `k = 3`.  In fact, the only value of `k` for which a tie would NOT be possible for a three category classification problem is `k = 2` (discounting the trivial scenario `k = 1`).  

So, what is a better way to break ties?  Using data sourced from http://archive.ics.uci.edu/ml/datasets/vertebral+column, I wanted to explore the default behavior of R's `knn` function, as well as introduce the `kknn` function, which is a weighted k-nearest neighbors method.

=========================================================================================

First off, we have to read the data into R.  This dataset is in the Attribute-Relation File Format (.arff), which provides us a chance to take a look at the `farff` package.

We will use the `readARFF` function to read the data into a dataframe, and then create an ID variable to identify observations.

```{r Load Data}
# install.packages('farff')
library(farff)
vert <- readARFF("column_2C_weka.arff")

# Create ID
ID <- rownames(vert)
vert <- cbind(ID = ID, vert)

```

KNN methods work best when variables are on similar scales, so let's normalize the variables, and add a suffix so we know these variables have been transformed.

```{r Normalize Variables}
# Create function for normalization
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}

# Normalize variables
vert.n <- as.data.frame(lapply(vert[2:7], normalize))

# Name new variables with a suffix '.n'
colnames(vert.n) <- paste0(colnames(vert.n), ".n")

# Put class variable back into normalized set
vert.n <- cbind(ID = ID, vert.n, class = vert$class)

# Remove unnecessary objects
rm(ID, normalize)

```


There are 6 explanatory variables here, but I really just want to look at two dimensions since it will be easier to visually see what is going on with these clusters. 

Let's use logistic regression to determine which two variables to focus on.  The p-values for `pelvic_radius.n` and `degree_spondylolisthesis.n` are both very low, so we will use those

```{r Variable Selection}

summary(glm(vert.n$class ~ 
              vert.n$pelvic_incidence.n +
              vert.n$pelvic_tilt.n +
              vert.n$lumbar_lordosis_angle.n +
              vert.n$sacral_slope.n +
              vert.n$pelvic_radius.n +
              vert.n$degree_spondylolisthesis.n,
            family = binomial(link = 'logit'),
            data = vert.n,
            control = list(maxit = 50)))

vert.n <- vert.n[c('ID', 'class', 'pelvic_radius.n', 'degree_spondylolisthesis.n')]

```

Let's go ahead and divide this data set into 70% train and 30% test, using our favorite random number.

```{r Train/Test}

# Assign variable for Train/Test designation
set.seed(8675309)
vert.n$train.0.test.1 <- sample(0:1,
                                nrow(vert.n),
                                replace = TRUE, 
                                prob = c(0.7, 0.3))

# Create Train/Test datasets
vert.n.train <- vert.n[vert.n$train.0.test.1 == 0,]
vert.n.test <- vert.n[vert.n$train.0.test.1 == 1,]
vert.n.train <- vert.n.train[c(-5)]
vert.n.test <- vert.n.test[c(-5)]
results <- vert.n.test[c('ID', 'class')]

```

We're not interested in necessarily choosing an optimal `k` here, I just want to choose an even number to force some ties.

The `use.all` option uses all equidistant observations instead of attempting to break a tie.

```{r KNN}
# Remove ID and class
vert.n.train.var <- vert.n.train[c(-1, -2)]
vert.n.test.var <- vert.n.test[c(-1, -2)]

# 2-nearest neighbor
selectk <- 2

# Predict a 2 neighbor solution
library(class)
knn <-  knn(vert.n.train.var, vert.n.test.var,
            vert.n.train$class,
            k = selectk, prob = TRUE, use.all = TRUE)

# Confusion Matrix of Results
table(knn, vert.n.test$class)

# Add reults to results dataset
results <- cbind(results, pred.uw = knn, prob.uw = attr(knn, 'prob'))

# Remove unnecessary objects
rm(knn, vert.n.train.var, vert.n.test.var)
```



```{r Weighted KNN}

# Predict a weighted 2 neighbor solution
library(kknn)
kknn <- kknn(class ~ ., vert.n.train, vert.n.test,
             k = selectk,
             distance = 1,
             kernel = "triangular")

# Append results
fit <- fitted(kknn)
results <- cbind(results, pred.w = fit, kknn$prob)
results$prob.w <- with(results, pmax(Normal, Abnormal))
results$Abnormal <- NULL
results$Normal <- NULL

# Table of results
table(fit, vert.n.test$class)

# Remove unnecessary objects
rm(fit, selectk)

```

Let's take a look at some of the observations which the models differed.

```{r Visual Exploration}

# Explore observations in which the weighted and unweighted models differ
diff <- results[which(results$pred.w==results$class & !(results$pred.w==results$pred.uw)),]
diff <- diff[c('ID')]
diff <- merge(vert.n.test, diff, by = 'ID')
diff$class <- 'Tie'

# Put these observations together with training data
train.tie <- rbind(vert.n.train, diff)

# Plot the training data with the classified points
plot(pelvic_radius.n ~ degree_spondylolisthesis.n,
     xlab = "Degree Spondylolisthesis",
     ylab = "Pelvic Radius",
     pch = c(16, 17, 18)[as.numeric(class)],  # different 'pch' types 
     main = "Vertebrae",
     col = c("red", "green", "blue")[as.numeric(class)],
     data = train.tie)
     legend('topright', 
            c('Abnormal', 'Normal', 'Unclassified'),
            col = c("red", "green", "blue"),
            pch = c(16, 17, 18))
# I want to zoom in to see what's going on
train.tie.m <- train.tie[which(train.tie$pelvic_radius.n < .6 & 
                               train.tie$pelvic_radius.n > .4 &
                               train.tie$degree_spondylolisthesis.n < .15),]

# Plot
plot(pelvic_radius.n ~ degree_spondylolisthesis.n,
     xlab = "Degree Spondylolisthesis",
     ylab = "Pelvic Radius",
     pch = c(16, 17, 18)[as.numeric(class)],  # different 'pch' types 
     main = "Vertebrae",
     col = c("red", "green", "blue")[as.numeric(class)],
     data = train.tie.m)
     legend('topright', 
            c('Abnormal', 'Normal', 'Unclassified'),
            col = c("red", "green", "blue"),
            pch = c(16, 17, 18))
with(subset(train.tie.m, class == 'Tie'), 
     text(degree_spondylolisthesis.n, pelvic_radius.n - .01, ID))

# Remove unnecessary objects
rm(train.tie.m)
```

Looking at Observation 30, it appears that 45 and 221 were used for classification, with `knn` choosing "Normal" based on a coin flip, while `kknn` chose "Abnormal" with a probability of ~ .69.

```{r Investigate Specific Point}

# I want to zoom in to see what's going on
train.tie.m <- train.tie[which(train.tie$pelvic_radius.n < .49 & 
                               train.tie$pelvic_radius.n > .42 &
                               train.tie$degree_spondylolisthesis.n < .026 &
                               train.tie$degree_spondylolisthesis.n > .015),]

# Plot
plot(pelvic_radius.n ~ degree_spondylolisthesis.n,
     xlab = "Degree Spondylolisthesis",
     ylab = "Pelvic Radius",
     pch = c(16, 17, 18)[as.numeric(class)],  # different 'pch' types 
     main = "Vertebrae",
     col = c("red", "green", "blue")[as.numeric(class)],
     data = train.tie.m)
     legend('topright', 
            c('Abnormal', 'Normal', 'Unclassified'),
            col = c("red", "green", "blue"),
            pch = c(16, 17, 18))
with(train.tie.m, 
     text(degree_spondylolisthesis.n, pelvic_radius.n, ID))
```
