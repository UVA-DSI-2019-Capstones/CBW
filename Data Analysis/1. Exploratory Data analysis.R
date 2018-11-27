library(dplyr)
library(ggplot2)

### Loading the data
load("text_data_sent.Rdata")

bess <- read.csv("CBW_Bess_tags_final.csv", stringsAsFactors = FALSE)

#"bio03_a626" "bio22_a628" "bio25_a628"
bess_title <- unique(bess[c("biographyID","collectionID","author", "personaName")])
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




#### HEAT MAP #####
#filter(text_all, author %in% top_authors$author),

text_all$time_slot <- cut(text_all$time,seq(0,1,0.05))
heat_map_time_slot <- group_by(text_all,author,time_slot) %>% 
              summarize(score_mean = mean(score,na.rm = TRUE))

ggplot(data = heat_map_time_slot,
       #, aes(x = time, y = score, col = author)) +
  aes(x = author, y = time_slot,fill = score_mean)) + geom_tile()+
  scale_fill_gradient(low = "red", high = "green")    
  
#### Ordering by positive and negative authors
library(stringdist)


authors <- sort(unique(text_all$author))
author_list <- list()

for(each_authors in authors){
  authors_check <- authors[!authors %in% each_authors]
  dist_vals <- stringdist(each_authors,authors_check)
  
  author_list[[each_authors]] <- authors_check[which(dist_vals < 7)]
  
  
}



### Map similar authors with different spellings


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


####### STARTS HERE ######## 

##sentiment with time for all the bios with the same author (for top 5 authors)


top_authors <-
  bess_title %>% group_by(author) %>% summarise(count = n()) %>%
  arrange(desc(count)) %>% head(5) %>% filter(author != "Unknown")


ggplot(data = filter(text_all, author %in% top_authors$author)
       , aes(x = time, y = score, col = author)) + geom_point()

##########

#sentiment with time for all the bios with the same persona 

top_persona <-
  text_all %>% group_by(personaName) %>% summarise(count = n()) %>%
  arrange(desc(count)) %>% head(3) %>% filter(personaName != "Unknown")


ggplot(data = filter(text_all, personaName %in% top_persona$personaName)
       , aes(x = time, y = score, col = personaName)) + geom_point()


##########################

#no of paras and sentiment categories 


text_data_sent$sentiment <- as.character(text_data_sent$sentiment)
text_data_sent_withoutNULL <- text_data_sent %>% filter(sentiment != "NULL")


ggplot(data.frame(table(text_data_sent_withoutNULL$sentiment)),aes(x = Var1,y=Freq)) + geom_bar(stat = "identity")



########## stacked bar chart

ggplot(data = filter(text_all, personaName %in% top_persona$personaName),
       aes(x = reorder(personaName, -ParagraphNo) , y = ParagraphNo , fill = sentiment)) + 
  geom_bar(stat = "identity")
