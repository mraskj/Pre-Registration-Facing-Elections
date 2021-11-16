library(data.table)
library(ggplot2)

setwd('/home/rask/Dropbox/Facing-Politics/')

df <- read.csv('data/all.csv', sep=',', header=T)
length(df[!is.na(df$imagelink), 'votes'])
df_reviewed <- read.csv('data/data.csv', sep=',', header=T)
mean(df[!is.na(df$imagelink), 'votes'], na.rm = T)
mean(df[is.na(df$imagelink), 'votes'], na.rm = T)



df = read.csv('models/men/gridsearch/gridsearch.csv', sep='\t', header=T)
df['X'] <- NULL
setDT(df)

#df[df$trainable=="True",]
#df[df$lr==1e-06,]

df$transfer <- ifelse(df$trainable=='True', 'Fine-tune', 'Feature Extraction')
df$name <- paste(df$transfer,', batch_size=', df$batch, ', lr=', df$lr, ', dropout=', df$dropout,sep = '')

df <- df[model=='VGG16']

mean_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=name
              ][,.SD[1], by=name]

#mean_df = mean_df[lr != 1e-04]

top <- mean_df[, .SD[order(mean, decreasing=T),][1:36]]

mean_df

# Plot that shows the top N best mean models
grid_plot <- ggplot(top, aes(x = reorder(name, mean), y=mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - sd,
                    ymax = mean + sd), width = 0) +
  labs(y='Mean balanced accuracy, ± std. dev.', x=NULL) +
  ylim(0,1)+
  scale_x_discrete(labels=rev(c(top$name))) + 
  coord_flip() + 
  theme_classic() + 
  scale_color_brewer(name=NULL, palette = "Set3") + 
  theme(legend.position = "bottom", 
        axis.text.y = element_text(size=8))

grid_plot

ggsave('gridsearch-men.jpg', grid_plot)

model_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=model
               ][,.SD[1], by=model
                 ][, c(1, 11:12)]
print(model_df)

batch_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=batch
               ][,.SD[1], by=batch
                 ][, c(1, 11:12)]
print(batch_df)

lr_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=lr
            ][,.SD[1], by=lr
              ][, c(1, 11:12)]
print(lr_df)

dropout_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=dropout
                 ][,.SD[1], by=dropout
                   ][, c(1, 11:12)]
print(dropout_df)

transfer_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=trainable
                  ][,.SD[1], by=trainable
                    ][, c(1, 11:12)]
print(transfer_df)


model_transfer_df <- df[, `:=`(mean = mean(acc), sd = sd(acc)), by=.(model, trainable)
               ][,.SD[1], by=.(model, trainable)
                 ][, c(1:2, 11:12)]
print(model_transfer_df)


# 
n1 <- c(0.66, 0.5, 0.64, 0.68)
n2 <- c(0.63, 0.5, 0.5, 0.52)
n3 <- c(0.7, 0.72, 0.69, 0.65)
mean_n <- sum(mean(n1) + mean(n2) + mean(n3))

mean(n1)/mean_n
mean(n2)/mean_n
mean(n3)/mean_n

2*2*3*3
