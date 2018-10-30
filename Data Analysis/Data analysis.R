library(dplyr)
library(ggplot2)

### Loading the data
load("text_data_sent.Rdata")

bess <- read.csv("CBW_Bess_tags_final.csv", stringsAsFactors = FALSE)
bess_title <- unique(bess[c("biographyID","collectionID","author")])
names(bess_title)[names(bess_title) == "collectionID"] <- "CollectionID"
names(bess_title)[names(bess_title) == "biographyID"] <- "BiographyID"
bess_title$key <- paste0(bess_title$CollectionID,"_",bess_title$BiographyID)

### Creating a unique id

text_data_sent$key <- paste0(text_data_sent$CollectionID,"_",text_data_sent$BiographyID)

avg_sentiment <- group_by(text_data_sent,key) %>% 
                  summarize(average_score = mean(score, na.rm = TRUE)) %>%
                    arrange(average_score) %>% left_join(bess_title)

View(avg_sentiment)


### Top Authors 

top_authors <-
bess_title %>% group_by(author) %>% summarise(count = n()) %>%
  arrange(desc(count)) %>% head(3) %>% filter(author != "Unknown")

top_authors$author

###

text_all <- left_join(text_data_sent,bess_title)
nrow(text_all)

ggplot(data = filter(text_all, author %in% top_authors$author)
       , aes(x = ParagraphNo, y = score, col = author)) + geom_line()


## standardizing

max_para_no <- group_by(text_all,CollectionID,BiographyID) %>% summarise(max(ParagraphNo))


text_all <- left_join( text_all,max_para_no)
text_all$time <- text_all$ParagraphNo/text_all$`max(ParagraphNo)`




ggplot(data = filter(text_all, author %in% top_authors$author)
       , aes(x = time, y = score, col = author)) + geom_point()


### 

overall <- group_by(text_all,time) %>% summarise(score = mean(score))

ggplot(data =overall
       , aes(x = time, y = score)) + geom_line()


ggplot(text_all,aes(time , score))+
  stat_summary(fun.data=mean_cl_normal) + 
  geom_smooth(method='lm')


## Logistic Regression 
lm_model <- lm(score ~ time, data = text_all)


#### Heat map

#x axis- authors (top 20 authors)
#y axis - Scores
