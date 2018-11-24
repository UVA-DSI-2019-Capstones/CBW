#Named Entity Extraction

#Setting the working directory
setwd('E:/Capstone IATH/Code')

#Loading required packages
library(readr)
library(httr)

#Loading the paragraph level content dataset of the bios
data = read_csv("textdata.csv")

#Extracting the text alone
text = data$ParagraphText

#Extracting keywords and named entities from IBM Natural Language Understanding API

#Function to access the api and get the required info
getanalysis = function(t){
  t <- shQuote(t)
  req <- GET(URLencode(paste("https://gateway.watsonplatform.net/natural-language-understanding/api/v1/analyze?version=2018-03-19&text=",t,"&features=entities",sep="")), 
             authenticate("be06fa62-7d91-4b85-ad4b-95f589edf37a", "yfAPi4cFATtF", type = "basic"))
  stop_for_status(req)
  q <- content(req)
  return(q)
}

#Getting the info and storing in the list
num <-length(text)
#Creating a list to store the info
e <- vector("list",num)
for(i in 1:num)
{
  tryCatch({
    p <- getanalysis(text[i])
    e[[i]] <- p
  }, error = function(error) {
    print(error)
    print(i)
  }
  )
}

save(e, file="entities.RData")

load("entities.RData")

getanalysis = function(t){
  t <- shQuote(t)
  req <- GET(URLencode(paste("https://gateway.watsonplatform.net/natural-language-understanding/api/v1/analyze?version=2018-03-19&text=",t,"&features=emotion",sep="")), 
             authenticate("0d29fae2-d6fa-45f6-b4c7-7f1c81cf2513", "DrSyJWIGBJeV", type = "basic"))
  stop_for_status(req)
  q <- content(req)
  return(q)
}

#Getting the info and storing in the list
num <-length(text)
#Creating a list to store the info
em <- vector("list",num)
for(i in 1:num)
{
  tryCatch({
    p <- getanalysis(text[i])
    em[[i]] <- p
  }, error = function(error) {
    print(error)
    print(i)
    
  }
  )
}
save(em, file="emotions.RData")

load("emotions.RData")




### Data Frame
entities <- lapply(e, "[" ,3)
entities_df <- data.frame(id_no = "", type = "", text = "", stringsAsFactors = FALSE)
count <- 0
start = Sys.time()
for(entity_no in 1:length(entities)){
  element_size <- length(entities[entity_no][[1]]$entities)
  list_element <- entities[entity_no][[1]]$entities
  
  entities_df_temp <- data.frame(id_no = "", type = "", text = "", stringsAsFactors = FALSE)
  
  if(element_size > 0){
    
  for(each in 1:element_size){
    
    entities_df_temp$id_no <- entity_no
    entities_df_temp$type <- list_element[[each]]$type
    entities_df_temp$text <- list_element[[each]]$text

    if(nrow(entities_df) > 0){
    entities_df <- rbind(entities_df,entities_df_temp)
    }else{
      entities_df <- entities_df_temp
    }
    
  }
  }
}

entities_df = entities_df[-1,]
data$id_no = row.names(data)

library(dplyr)

data_entities = left_join(entities_df, data, by = id_no)
save(data_entities, file="data_entities.RData")

sadness = c()
joy = c()
fear = c()
disgust = c()
anger = c()

sadness <- unlist(sapply(em, function(x)ifelse(is.null(x),NA,x$emotion$document$emotion$sadness)))
joy <- unlist(sapply(em, function(x)ifelse(is.null(x),NA,x$emotion$document$emotion$joy)))
fear <- unlist(sapply(em, function(x)ifelse(is.null(x),NA,x$emotion$document$emotion$fear)))
disgust <- unlist(sapply(em, function(x)ifelse(is.null(x),NA,x$emotion$document$emotion$disgust)))
anger <- unlist(sapply(em, function(x)ifelse(is.null(x),NA,x$emotion$document$emotion$anger)))

data_emotions <-cbind(data, sadness, joy, fear, disgust, anger)

save(data_emotions, file = "data_emotions.Rdata")

load("data_sentiment.RData")

### data updated - concat rows with same combos of C Id, B Id and para number

data_final <- group_by(data, CollectionID,BiographyID, ParagraphNo) %>% summarise(ParagraphText = paste0(ParagraphText, collapse = " "))
write_csv(data_final, "textdata2.csv")


data_emotions_sentiment = data_emotions
data_emotions_sentiment$score = text_data_sent$score
data_emotions_sentiment$sentiment = text_data_sent$sentiment

data_entities$type_text <- paste0(data_entities$type,"_",data_entities$text)
data_entities$value <- 1



data_entities_long <- dcast(data_entities, id_no + ParagraphNo + ParagraphText + CollectionID + BiographyID  ~ type_text,
                            value.var = "value", fun.aggregate = sum)

### Group by


data_entities_backup <- data_entities
data_entities$ParagraphText <- NULL

data_entities_group <-

data_entities %>% group_by(ParagraphNo,CollectionID,BiographyID,type) %>%
                
                  summarize(type_all = paste0(sort(unique(text)),collapse = ", "))


data_entities_association <- 
  spread(data_entities_group, key = type, value = type_all)



data_with_entities_final <- left_join(data_final, data_entities_association)

nrow(data_final)
nrow(data_with_entities_final)

data_emotions_sentiment_backup = data_emotions_sentiment

data_emotions_sentiment$ParagraphText = NULL
data_emotions_sentiment$id_no = NULL
data_emotions_sentiment$sentiment = NULL

data_emo_sent = data_emotions_sentiment %>% group_by(CollectionID, BiographyID, ParagraphNo) %>%
  
  summarize(sadness = mean(sadness), joy = mean(joy), fear = mean(fear), disgust = mean(disgust), anger = mean(anger), score = mean(score))

data_emo_sent$sentiment = character(nrow(data_emo_sent))
data_emo_sent$sentiment[data_emo_sent$score == 0] = "neutral"
data_emo_sent$sentiment[data_emo_sent$score > 0] = "positive"
data_emo_sent$sentiment[data_emo_sent$score < 0] = "negative"


save(data_with_entities_final, file = "data_entities.Rdata")
save(data_emo_sent, file = "data_emotions_sentiment.Rdata")

text_features = left_join(data_emo_sent, data_with_entities_final)
save(text_features, file = "text_features.Rdata")
write_csv(text_features, "text_features.csv")

