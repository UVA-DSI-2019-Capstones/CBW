################## IMPORTING REQUIRED PACKAGES ########################
import pandas as pd
#import numpy as np
#import os
import string
from nltk.corpus import stopwords
from collections import Counter 
from itertools import combinations, groupby

# functions to detect bi-grams
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import word_tokenize #Tokenize words
from nltk.stem import WordNetLemmatizer #Lemmatization

import nltk
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')




#################### READING THE CONTENTS OF EACH PARAGRAPH ####################

file_path = 'C:\\Users\\arvra\\Documents\\UVa files\\Classes\\Fall_18\\Capstone Project\\'
textdatanew = pd.read_csv(file_path+"textdatanew.csv",encoding='ISO-8859-1')
textdatanew.head()

text_data = textdatanew


text_data['Unique_id'] = text_data.CollectionID+"_"+text_data.BiographyID+"_"+text_data.ParagraphNo.apply(str)
unique_id_col = 'Unique_id'
text_col = 'ParagraphText'
##################### Data Preprocessing function #################################


def text_pre_process(text_data,unique_id_col, text_col,lemma = True,bi_grams_detect = True):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    print("Tokenizing words...")
    
    if(lemma):
        text_data['Word_split'] = \
        text_data[text_col].apply(lambda x: [wordnet_lemmatizer.lemmatize(words) for words in word_tokenize(x) if words not in string.punctuation])
        
    else:
        text_data['Word_split'] = \
        text_data[text_col].apply(lambda x: [words for words in word_tokenize(x) if words not in string.punctuation])
      
    
    
    ############################## Bi-gram detector ###########################
    
    if(bi_grams_detect):
        print("Extracting Bi-grams...")
        word_split_sent = text_data.Word_split.values
        bigram = Phrases(word_split_sent, delimiter=b' ',min_count=10)
        bigram_phraser = Phraser(bigram)
     
        ## Including the text with bi-gram
        text_data['Word_split_new'] = text_data.Word_split.apply(lambda x: bigram_phraser[x])
  
    else:
        text_data['Word_split_new'] = text_data.Word_split
  
  
    ############################# Extracting the POS tagging ###########################
    print("Extracting the POS tagging...")
    text_data['Word_split_POS'] = text_data.Word_split_new.apply(lambda x: nltk.pos_tag(x))
    
    ############################## Rearranging the data ################################
    print("Preparing Data for Exporting...")
    text_stack = text_data.apply(lambda x: pd.Series(x['Word_split_new']), axis=1).stack().reset_index(level=1, drop=True)
    text_stack_with_pos = text_data.apply(lambda x: pd.Series(x['Word_split_POS']), axis=1).stack().reset_index(level=1, drop=True)
    
    text_pos = text_stack_with_pos.apply(lambda x:x[1])
    
    text_stack.name = 'Word_splits'
    text_pos.name = 'POS'
    
    text_stack_join = pd.concat([text_stack,text_pos],axis = 1)
    
    stack_cols = [unique_id_col]
    text_data_words = text_data[stack_cols].join(text_stack_join)
    
    text_data_words['char_len'] = text_data_words.Word_splits.apply(lambda x: len(x))

    return(text_data_words)
    
############################END OF FUNCTION ###########################################################    
    
   
def find_stop_words(text_data_words):    
    ############ Identifying and Removing Stop words ######################################
    
    ### Tokenized Words
    word_counts = Counter(text_data_words.Word_splits.apply(lambda x: x.lower()))
    most_common_words = [each[0] for each in word_counts.most_common(100)]
    
    
    text_data_words['Word_freq'] = text_data_words.Word_splits.apply(lambda x: word_counts[x.lower()])
    
    stop_words_list = stopwords.words('english')

    ### Adding the most common words
    stop_words_list.extend(most_common_words)
    
    ## Adding the least frequenct words
    less_freq_words = text_data_words[text_data_words.Word_freq < 10].Word_splits.str.lower().values
    stop_words_list.extend(less_freq_words)
    
    #### Removing words with less than or equal to two characters and making sure they are not numeric
    words_2_char = text_data_words[text_data_words.char_len < 3].Word_splits.unique().tolist()
    words_2_char.extend(words_2_char)
    
    word_2_char_to_remove = [each for each in words_2_char if ~each.isdigit()]
    stop_words_list.extend(word_2_char_to_remove)
    
    return(stop_words_list)
############################END OF FUNCTION ###########################################################        
    
    
################################################################################################################
################################ ASSOCIATION RULE FUNCTIONS ####################################################
################################################################################################################
    
# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    #print(order_item)
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]   
        

def association_rules(order_item, min_support):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)
        
################################################################################################################
################################# END OF ASSOCIATION RULE FUNCTIONS#############################################
################################################################################################################
        



### Calling all the created functions

# Pre-processing
text_data_words = text_pre_process(text_data,unique_id_col, text_col,lemma = True,bi_grams_detect = True)

## Removing stop words based on the find_stop_words function
stop_words_list = find_stop_words(text_data_words)
Text_final = text_data_words[~text_data_words.Word_splits.str.lower().isin(stop_words_list)]

## Lower case of words
Text_final['Words_clean'] = Text_final.Word_splits.apply(lambda x: x.lower()).values
    

# Creating Data to be read by Association rule
Words = Text_final[['Unique_id','Words_clean']].drop_duplicates().set_index('Unique_id')['Words_clean'].rename('item_id')


# Calling Association Rule function
rules = association_rules(Words, 0.01)  


## Viewing the obtained result
rules[rules.item_A == 'france']







