library(data.table)

# set working directory
setwd('/home/rask/Dropbox/Facing-Politics/')

# read data and set as datatable object
df <- read.csv('data/data.csv', sep=',', header=T)
setDT(df)

# compute table
table <- table(df$gender, df$elected)

round(df[gender==0, sum(elected)/.N],3)*100    # Elected women
round(df[gender==1, sum(elected)/.N],3)*100    # Elected men

round(df[gender==0, (.N - sum(elected))/.N]*100, 1) # Unelected women
round(df[gender==1, (.N - sum(elected))/.N]*100,1)  # Elected women