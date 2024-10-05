
# Library Setup -----------------------------------------------------------------

# Install needed packages
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if (!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

# Load needed packages
library(tidyverse)
library(caret)
library(gridExtra)
library(rpart.plot)

# Setup Dataset -----------------------------------------------------------

# Link to the original Dataset: https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data
# The dataset will be downloaded from the Github repository of this project.

data <- "./nearest-earth-objects.csv"
git_link <- "https://raw.githubusercontent.com/Fuechselbruezel/HarvardX-Capstone-CYO/refs/heads/main/nearest-earth-objects.csv"

# Check if dataset already exists, otherwise download from Github
if (!file.exists(data)) {
      # Use try Catch to handle errors
      tryCatch({
        # download the file
        download.file(git_link, data, method = "curl")
        print("File downloaded successfully")
      }, error = function(e) {
        # Stop execution and print error
        stop(cat("Error downloading the file:", e$message, "\n"))
      })
   }

#Load the Dataset
neo <- read_csv(data)
#remove link and file reference
rm(data, git_link)

# Analysis ----------------------------------------------------------------

# Set our goal accuracy to later compare it to the models accuracy 
goal_accuracy <- 0.95

# Omit NA-Values, remove orbiting_body and name column from dataset
neo <- na.omit(neo)
neo$orbiting_body <- NULL
neo$name <- NULL

# Calculate correlation matrix
neo %>% 
  cor() %>% 
  as.data.frame() %>%
  select(is_hazardous)

# Create Visualization subset
# using seed for reproducibility, set p to 0.005 to create 0.5% subset
set.seed(007, sample.kind = "Rounding")
p_vis <- 0.005
index <- createDataPartition(y = neo$is_hazardous, p = p_vis, list = FALSE)
neo_vis <- neo[index,]
rm(index)

# Create Density plot of absolute_magnitude
neo_vis %>% ggplot(aes(absolute_magnitude, fill = is_hazardous, alpha = 0.1)) +
  geom_density() + 
  theme(legend.position = "top") +
  guides(alpha = "none")

# create scatter plots of absolute_magnitude vs miss_distance and relative_velocity
plot1 <- neo_vis %>% ggplot(aes(miss_distance, absolute_magnitude, color = is_hazardous, alpha = 0.1)) +
  geom_point() + 
  theme(legend.position = "top") +
  guides(alpha = "none")

plot2 <- neo_vis %>% ggplot(aes(relative_velocity, absolute_magnitude, color = is_hazardous, alpha = 0.1)) +
  geom_point() + 
  ylab(NULL) +
  theme(legend.position = "top") +
  guides(alpha = "none")

grid.arrange(plot1, plot2, nrow = 1)
rm(plot1, plot2)


## Measuring Accuracy ----------------------------------------------------------------

accuracy <- function(y_actual, y_predicted) {
  mean(ifelse(y_actual == y_predicted, 1, 0))
}


## Train/Test set ----------------------------------------------------------------

# using seed for reproducibility
# split dataset into training and test set by 80%, 20% respectively
set.seed(1234, sample.kind = "Rounding")
p <- 0.2
index <- createDataPartition(y = neo$is_hazardous, p = p, list = FALSE)
neo_train <- neo[-index,]
neo_test <- neo[index,]
rm(index)


## A Trivial Approach ----------------------------------------------------------------

# Calculate the mean and accuracy
mean <- mean(neo_train$is_hazardous)
acc_trivial <- accuracy(neo_test$is_hazardous, round(mean))

# Plot the percentages of the dataset
neo_train %>% ggplot(aes(is_hazardous)) + 
  geom_bar(aes(y = (after_stat(count))/sum(after_stat(count)))) + 
  geom_hline(yintercept = mean) + 
  xlab("Is Hazardous") + 
  ylab("Percentage")

# Create result collection and present the result
results <- tibble(Model = "Trivial", Accuracy = acc_trivial, Goal = acc_trivial > goal_accuracy)
results


## Modelbuilding ----------------------------------------------------------------

### Rpart ----------------------------------------------------------------

# Train the Rpart algorithm on trainset and use it to predict the values for the testset
model_rpart <- train(factor(is_hazardous) ~ ., data = neo_train, method = "rpart")
rpart_pred <- predict(model_rpart, neo_test)

# Plot the Rpart model
rpart.plot(model_rpart$finalModel)

# Calculate Accuracy of the Rpart model
acc_rpart <- accuracy(rpart_pred, neo_test$is_hazardous)

# Plot the Feature Importance of the Rpart model
plot(varImp(model_rpart))

# Add to result collection and present
results <- bind_rows(results, tibble(Model = "Rpart", Accuracy = acc_rpart, Goal = acc_rpart > goal_accuracy))
results


### Gradient Boost ----------------------------------------------------------------

# Setup Train Control as 10-fold Cross validation.
# This will be used for gbm and Random Forest
trControl <- trainControl(method = "cv", number = 10)

# Setup tune grid to find best tuned parameters for gbm.
# Evaluation threw subset used to plot relationships earlier
gbm_grid <- expand.grid(interaction.depth = c(3:6), n.trees = (1:30)*50, shrinkage = 0.1, n.minobsinnode = 10)

# Evaluate best tuned parameters
model_gbm_tune <- train(factor(is_hazardous) ~ ., data = neo_vis, method = "gbm", tuneGrid = gbm_grid, verbose = FALSE)

# Plot the tuning proccess and present best parameters.
gbm_best <- model_gbm_tune$bestTune
ggplot(model_gbm_tune, highlight = TRUE)
gbm_best

# using seed for reproducibility, train the gbm algorithm on the trainset and predict values for testset
set.seed(522, sample.kind = "Rounding")
model_gbm <- train(factor(is_hazardous) ~ ., data = neo_train, method = "gbm", trControl = trControl, tuneGrid = gbm_best, verbose = FALSE)
gbm_pred <- predict(model_gbm, neo_test)

# Calculate Accuracy of final gbm model
acc_gbm <- accuracy(gbm_pred, neo_test$is_hazardous)

# Add to result collection and present
results <- bind_rows(results, tibble(Model = "Gradient Boost", Accuracy = acc_gbm, Goal = acc_gbm > goal_accuracy))
results


### Random Forest ----------------------------------------------------------------

# Setup tune grid to find best tuned parameters for Random Forest.
# Evaluation threw subset used to plot relationships earlier
rf_grid <- data.frame(mtry = seq(1,6))

# Evaluate best tuned parameters
model_rf_tune <- train(factor(is_hazardous) ~ ., data = neo_vis, method = "rf", tuneGrid = rf_grid)
rf_best <- model_rf_tune$bestTune

# using seed for reproducibility
set.seed(568, sample.kind = "Rounding")

# Train the Random Forest algorithm on the trainset and predict values for testset
model_rf <- train(factor(is_hazardous) ~ ., data = neo_train, 
                  method = "rf", trControl = trControl, tuneGrid = rf_best)
rf_pred <- predict(model_rf, neo_test)

# Calculate Accuracy of final Random Forest model
acc_rf <- accuracy(rf_pred, neo_test$is_hazardous)

# Add to result collection and present
results <- bind_rows(results, tibble(Model = "Random Forest", Accuracy = acc_rf, Goal = acc_rf > goal_accuracy))
results


# Results ----------------------------------------------------------------

# Present final results
results

# Present Feature Importance of final model
plot(varImp(model_rf))
