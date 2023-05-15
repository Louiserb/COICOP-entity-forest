library(readr)
library(tidyverse)
library(ggplot2)
library(DescTools)
library(tm)

# read in data
data(df) ## all data
data(NORWAY_COICOP) ## examples of COICOP codes

# data cleaning
df %>% 
  mutate(across(where(is.character), tolower)) -> df

my_stopwords <- c('pk', 'ca', 'gr', 'kk', 'pos', 'ex', 'va', 'av', 'alu', 'for', 'vac', ' vv', 'plu', 'pe', 'mk', 'kll', 'co', 'cs', 'ank', 'dato', 'parcel', 'containernummer')
stopwords <- paste0("\\b(", paste(my_stopwords, collapse="|"), ")\\b")

## remove defined stopwords
df %>%
  mutate(cleanText = gsub(stopwords, "", cleanText)) %>%
  select(cleanText, subclass) -> df

## remove numbers
df$cleanText <- gsub("\\d+", "", df$cleanText)

## remove special characters
df$cleanText <- gsub("[[:punct:]]", "", df$cleanText)

## remove newline characters
df$cleanText <- gsub("\\n", "", df$cleanText)

## remove single characters
df$cleanText <- gsub("\\b\\w{1}\\b", "", df$cleanText)


# 1. Change subclass of non-distinct to COICOP NORWAY:
matched_words <- match(df$cleanText, NORWAY_COICOP$cleanText)
df$subclass[!is.na(matched_words)] <- NORWAY_COICOP$subclass[matched_words[!is.na(matched_words)]]


# 2. If classified to multiple subclasses, chose the one which is most used

df_2 <- df

## group the data by descriptions and COICOP code- count descriptions in groups
counted <- df_2 %>%
  group_by(cleanText, subclass) %>%
  summarize(n = n()) %>%
  ungroup()

## calculate the variance of groups
variance <- counted %>%
  group_by(cleanText) %>%
  summarize(var = var(n)) %>%
  filter(var > 0) %>%
  select(cleanText)

## use the subset with non-zero variance to filter out duplicates
df_subset <- df_2 %>%
  filter(cleanText  %in% variance$cleanText)


## again- group data by description and COICOP code - count descriptions in groups
counted2 <- df_subset %>%
  group_by(cleanText, subclass) %>%
  summarize(n = n()) %>%
  ungroup()


## find the group with the highest count and reclassify all instances of to that code
max_group <- counted2 %>%
  group_by(cleanText) %>%
  slice(which.max(n)) %>%
  select(cleanText, subclass)

df_reclassified <- df_subset %>%
  left_join(max_group, by = "cleanText") %>%
  select(cleanText, subclass.y) %>%
  rename(subclass = subclass.y)


# 3. Delete duplicates that are left

df_conclusive <- df

df_delete <- original_duplicated_after2 %>%
  select(cleanText,subclass)

df_conclusive <- anti_join(df_conclusive, df_delete, by = c("cleanText"))

