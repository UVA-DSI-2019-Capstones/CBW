#Named Entity Extraction

#Setting the working directory
setwd('/Users/user/Documents/CBW/Data\ Preparation')

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
  req <- GET(URLencode(paste("https://gateway.watsonplatform.net/natural-language-understanding/api/v1/analyze?version=2018-03-19&text=",t,"&features=sentiment",sep="")), 
             authenticate("591a0490-8aff-40ed-bb02-0bc51a5636b1", "VANeK0dZsWdR", type = "basic"))
  stop_for_status(req)
  q <- content(req)
  return(q)
}

#Getting the info and storing in the list
num <-length(text)
#Creating a list to store the info
sent <- vector("list",num)
for(i in 1:num)
{
  tryCatch({
    p <- getanalysis(text[i])
    sent[[i]] <- p
  }, error = function(error) {
    print(error)
    print(i)
  }
  )
}

save(sent, file="sent.RData")

load("sent.RData")

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





# a = e[[1]]$entities
# length(a)
# for(i in 1:length(a)){
#   type[i] = a[[i]]$type
#   entity[i] = a[[i]]$text
# }


sentiment_score<-c()
sentiment<-c()

##SCORE
score <- 
  unlist(sapply(sent, function(x)ifelse(is.null(x),NA,x$sentiment$document$score)))

### SENTIMENT
sentiment <- 
  unlist(sapply(sent, function(x)ifelse(is.null(x),"NULL",x$sentiment$document$label)))



text_data_sent<-cbind(data, score, sentiment)

save(text_data_sent,file = "text_data_sent.Rdata")
