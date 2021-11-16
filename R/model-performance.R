library(ggplot2)
library(data.table)

setwd('/home/rask/Dropbox/Facing-Politics/')

df_women <- read.csv('model-performance_women.csv', sep=',', header=T)
df_women['X'] <- NULL
setDT(df_women)
df_women[, gender := 'women']

df_men <- read.csv('model-performance_men.csv', sep=',', header=T)
df_men['X'] <- NULL
setDT(df_men)
df_men[, gender := 'men']

df <- rbind(df_men, df_women)

# compute median by gender
median_val_bal_acc <- df[,median(val_bal_acc) ,by=gender]

# make plot
p <- ggplot(df, aes(x=val_bal_acc, fill=gender)) +
  geom_density(alpha=.5, color='grey30') + 
  geom_vline(data=mean_val_bal_acc, aes(xintercept=V1, color=gender),
             linetype="dashed", show.legend = F) + 
  labs(x='Validation balanced accuracy', title = 'Distribution of model performances') + 
  theme_light() + 
  theme(legend.position=c(.9,.8),legend.justification="right",
        legend.background = element_rect(color = NA),
        legend.key=element_blank(), legend.title=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        plot.title = element_text(hjust = 0.5))
ggsave('model_performance.jpg', p)


## VALIDATION WOMEN

# MAIN model
val <- sort(df_women$val_bal_acc)[10]
df_women[val_bal_acc==val]

# worst model
val <- sort(df_women$val_bal_acc)[1]
df_women[val_bal_acc==val]

# best model
val <- sort(df_women$val_bal_acc)[nrow(df_women)]
df_women[val_bal_acc==val]

# average
mean(df_women$val_bal_acc)


