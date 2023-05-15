import pandas as pd
import re
import warnings
from nltk import ngrams
from collections import Counter
from collections import defaultdict
import numpy as np
import random

warnings.filterwarnings('ignore')

####### BUILD ENTITY FOREST #######

def build(X_train, checklist):
  """
  PART 1: Root term based on SSB item description exampl.
  PART 2: extract root terms based on most frequent words and bigrams. 
  PART 3: Extract tree when root is given in forest2. 
  PART 4: Join the datasets and get full dataset as an entity forest.

  Args:
    X_train (list): dataframe with item text and COICOP-5 codes.
    checklist(list): item description examples from SSB

  Returns:
    entity forest (list): list of dictionaries.

  """

  shoppinglist = X_train
  coicop = checklist

  # remove instances of text with more than two words in checklist
  coicop['word_count'] = coicop['word'].apply(lambda x: len(re.findall('\w+', x)))
  checklist = coicop[coicop['word_count'] <= 2].drop(columns=['word_count'])

  """
    PART 1: Root term based on SSB item description example.
  """
  not_matched = pd.DataFrame()

  def extract_words(description, class_name):
      root = ''
      before_1 = ''
      before_2 = ''
      before_3 = ''
      before_4 = ''
      before_5 = ''
      after_1 = ''
      after_2 = ''
      after_3 = ''
      after_4 = ''
      after_5 = ''
      
      # go through the checklist to find a match
      for index, row in checklist[checklist['subclass'] == class_name].iterrows():
          word = row['word']
          if re.search(r'\b' + re.escape(word) + r'\b', description):
              root = word
              # get words before root
              before_words = re.findall('\w+', description[:description.index(root)])
              if before_words:
                  before_1 = before_words[-1]
              if len(before_words) > 1:
                  before_2 = before_words[-2]
              if len(before_words) > 2:
                  before_3 = before_words[-3]
              if len(before_words) > 3:
                  before_3 = before_words[-4]
              if len(before_words) > 4:
                  before_3 = before_words[-5]            

              # get words after root
              after_words = re.findall('\w+', description[description.index(root) + len(root):])
              if after_words:
                  after_1 = after_words[0]
              if len(after_words) > 1:
                  after_2 = after_words[1]
              if len(after_words) > 2:
                  after_3 = after_words[2]
              if len(after_words) > 3:
                  after_3 = after_words[3]
              if len(after_words) > 4:
                  after_3 = after_words[4]
              
              break

      return root, before_1, before_2, before_3, before_4, before_5, after_1, after_2, after_3, after_4, after_5



  forest = pd.DataFrame(columns=['root', 'before_1', 'before_2', 'before_3', 'before_4', 'before_5', 'after_1', 'after_2', 'after_3', 'after_4', 'after_5', 'subclass', 'description']) # Add 'description' as a column in the DataFrame

  # go through the shoppinglist and extract words
  for index, row in shoppinglist.iterrows():
      class_name = row['subclass']
      description = row['cleanText']
      root, before_1, before_2, before_3, before_4, before_5, after_1, after_2, after_3,  after_4, after_5 = extract_words(description, class_name)
      # add values to the forest dataset
      if root:
          forest = forest.append({'root': root, 'before_1': before_1, 'before_2': before_2, 'before_3': before_3, 'before_4': before_4,'before_5': before_5, 'after_1': after_1, 'after_2': after_2, 'after_3': after_3, 'after_4': after_4, 'after_5': after_5, 'subclass': class_name, 'description': description}, ignore_index=True) # Add 'description' as a value in the DataFrame
      else:
          not_matched = not_matched.append({'cleanText': description, 'subclass': class_name}, ignore_index=True)



  """
  PART 2: Extract root terms based on most frequent words and bigrams. 
  """

  df = not_matched

  # dictionary with the most frequent words for each class
  most_frequent_words = {}
  for class_ in df['subclass'].unique():
      words = [word for text in df[df['subclass'] == class_]['cleanText'] for word in text.split()]
      most_frequent_words[class_] = [x[0] for x in Counter(words).most_common()]

  # dictionary with the most frequent bigrams for each class
  most_frequent_bigrams = {}
  for class_ in df['subclass'].unique():
      bigrams = [bigram for text in df[df['subclass'] == class_]['cleanText'] for bigram in ngrams(text.split(), 2)]
      most_frequent_bigrams[class_] = [x[0] for x in Counter(bigrams).most_common()]


  word_count = {}
  for class_, words in most_frequent_words.items():
      for word in words:
          if word not in word_count:
              word_count[word] = {}
          word_count[word][class_] = most_frequent_words[class_].count(word)

  for word, counts in word_count.items():
      classes = [class_ for class_, count in counts.items() if count > 0]
      if len(classes) > 1:
          max_count = 0
          max_class = ''
          for class_, count in counts.items():
              if count > max_count:
                  max_count = count
                  max_class = class_
          for class_, words in most_frequent_words.items():
              if class_ != max_class and word in words:
                  most_frequent_words[class_].remove(word)


  bigram_count = {}
  for class_, bigrams in most_frequent_bigrams.items():
      for bigram in bigrams:
          if bigram not in bigram_count:
              bigram_count[bigram] = {}
          bigram_count[bigram][class_] = most_frequent_bigrams[class_].count(bigram)

  for bigram, counts in bigram_count.items():
      classes = [class_ for class_, count in counts.items() if count > 0]
      if len(classes) > 1:
          max_count = 0
          max_class = ''
          for class_, count in counts.items():
              if count > max_count:
                  max_count = count
                  max_class = class_
          for class_, bigrams in most_frequent_bigrams.items():
              if class_ != max_class and bigram in bigrams:
                  most_frequent_bigrams[class_].remove(bigram)
  single_word = []
  forest2 = pd.DataFrame(columns=['cleanText', 'root', 'subclass'])

  for index, row in df.iterrows():
      text = row['cleanText']
      class_ = row['subclass']
      
      root = None
      for word in most_frequent_words[class_]:
          if word in text.split() and word not in forest['root'].values:
              root = word
              break

      if root is None:
          words = text.split()
          for i in range(len(words) - 1):
              word1, word2 = words[i], words[i+1]
              bigram = f"{word1} {word2}"
              value_to_find = (word1, word2)
              if (value_to_find in most_frequent_bigrams[class_]) and (value_to_find not in forest['root'].values):
                  root = bigram
                  break

      if root is None:
          single_word.append(text)

      forest2 = forest2.append({'cleanText': text, 'root': root, 'subclass': class_}, ignore_index=True)


  def remove_none_lines(data):
      rows_with_none = data.isna().any(axis=1)
      data = data[~rows_with_none]

      return data
  # remove all lines with no values
  forest2 = remove_none_lines(forest2)

  not_matched2 = pd.DataFrame()

  # forest2 to pandas DataFrame
  forest2 = pd.DataFrame(forest2)

  """
  PART 3: Extract tree when root is given in forest2. 
  """
  def extract_words2(description, class_name):
      root = ''
      before_1 = ''
      before_2 = ''
      before_3 = ''
      before_4 = ''
      before_5 = ''
      after_1 = ''
      after_2 = ''
      after_3 = ''
      after_4 = ''
      after_5 = ''
      
      for index, row in forest2[forest2['subclass'] == class_name].iterrows():
          word = row['root']
          if not word:
            continue
          if re.search(r'\b' + re.escape(word) + r'\b', description):
              root = word
              before_words = re.findall('\w+', description[:description.index(root)])
              if before_words:
                  before_1 = before_words[-1]
              if len(before_words) > 1:
                  before_2 = before_words[-2]
              if len(before_words) > 2:
                  before_3 = before_words[-3]
              if len(before_words) > 3:
                  before_3 = before_words[-4]
              if len(before_words) > 4:
                  before_3 = before_words[-5]            

              after_words = re.findall('\w+', description[description.index(root) + len(root):])
              if after_words:
                  after_1 = after_words[0]
              if len(after_words) > 1:
                  after_2 = after_words[1]
              if len(after_words) > 2:
                  after_3 = after_words[2]
              if len(after_words) > 3:
                  after_3 = after_words[3]
              if len(after_words) > 4:
                  after_3 = after_words[4]
              
              break

      return root, before_1, before_2, before_3, before_4, before_5, after_1, after_2, after_3, after_4, after_5


  forest3 = pd.DataFrame(columns=['root', 'before_1', 'before_2', 'before_3', 'before_4', 'before_5', 'after_1', 'after_2', 'after_3', 'after_4', 'after_5', 'subclass', 'description'])

  for index, row in forest2.iterrows():
      class_name = row['subclass']
      description = row['cleanText']
      root, before_1, before_2, before_3, before_4, before_5, after_1, after_2, after_3,  after_4, after_5 = extract_words2(description, class_name)
      if root:
          forest3 = forest3.append({'root': root, 'before_1': before_1, 'before_2': before_2, 'before_3': before_3, 'before_4': before_4,'before_5': before_5, 'after_1': after_1, 'after_2': after_2, 'after_3': after_3, 'after_4': after_4, 'after_5': after_5, 'subclass': class_name, 'description': description}, ignore_index=True)
      else:
        not_matched2 = not_matched2.append({'cleanText': description, 'subclass': class_name}, ignore_index=True)



  """
  PART 4: Join the datasets and get full dataset as an entity-forest.
  """

  # columns to keep and rearrange
  forest_sub1 = forest.loc[:, ['description','subclass', 'before_5', 'before_4', 'before_3', 'before_2', 'before_1','root','after_1','after_2', 'after_3', 'after_4', 'after_5']]

  forest_sub3 = forest3.loc[:, ['description', 'subclass', 'before_5', 'before_4', 'before_3', 'before_2', 'before_1','root','after_1','after_2', 'after_3', 'after_4', 'after_5']]

  # concatinate the two dataframes
  total_forest = pd.concat([forest_sub1, forest_sub3])



  from collections import defaultdict

  # create dictionary of item descriptions
  merged_dicts = defaultdict(lambda: {'root': [], 'terms': [], 'coicop_code': []})


  x_vec = []
  for index, row in total_forest.iterrows():
      description = row['description']
      root = row['root']
      branch = []
      terms = []

      # add non-root terms to branch list
      for i in range(1, 6):
          if row[f'before_{i}']:
              branch.append(row[f'before_{i}'])
          if row[f'after_{i}']:
              branch.append(row[f'after_{i}'])
      
      # add root term to terms list
      if root:
          terms.append(root)

      # add non-root terms to terms list
      terms.extend(branch)

      coicop_code = row['subclass']

      # get the dictionary for the current row
      entity_dict = {
          #'description' : description,
          'root': [root] if root else [],
          #'branch': branch,
          'terms': terms,
          'coicop_code': coicop_code
      }


      # merge dictionary with the same root and coicop code
      key = (tuple(entity_dict['root']), tuple(entity_dict['coicop_code']))
      merged_dicts[key]['root'].extend(entity_dict['root'])
      merged_dicts[key]['terms'].extend(entity_dict['terms'])
      merged_dicts[key]['coicop_code'].extend(entity_dict['coicop_code'])
      #  get merged dictionaries back to a list
      entity_forest = [{'root': merged_dict['root'], 'terms': list(set(merged_dict['terms'])), 'coicop_code': merged_dict['coicop_code']} for merged_dict in merged_dicts.values()]
      # add the dictionary to the x_vec list
      x_vec.append(entity_dict)
      
      entity_forest = x_vec
    
  return entity_forest


 ####### CLASSIATION #######

def match_for_COICOP_5_code(x, x_list, alpha, beta):
    """
    get the match score for a given item description 'x' against a list of
    item descriptions 'x_list' by using agreement or not.
    
    Args:
        x (str): item description to be classified.
        x_list (list): list of item descriptions for COICOP-5 codes.
        alpha (float): score for root term match.
        beta (float): score for branch term match.
    
    Returns:
        match score for the item description 'x'.
    """

    match_score = 0
    x_terms = x.split()
    
    # if match on root bi-gram
    for i in range(len(x_terms)-1):
        bi_gram = x_terms[i] + ' ' + x_terms[i+1]
        if bi_gram in x_list['root']:
            match_score += alpha
            if any(term in x_list['terms'] for term in x_terms):
                match_score += beta
    
    # if match on root
    for xi in x_terms:
        if xi in x_list['root']:
            match_score += alpha
            if any(term in x_list['terms'] for term in x_terms):
                match_score += beta
    # If not match on root - only branch
        elif xi in x_list['terms']:
          match_score += beta

    return match_score



def classify_item_description(x, entity_forest, alpha, beta):
    """
    classify an item description 'x' to 5 digit COICOP code using the entity forest.
    
    Args:
        x (str): item description to be classified.
        entity_forest (list): list of dictionaries representing the entity-forest.
        alpha (float): score for root term match.
        beta (float): score for branch term match.
    
    Returns:
        5 digit COICOP code with the highest match score for the item description 'x'.
    """
    max_match_score = 0
    chosen_code = None
    for entity in entity_forest:
        match_score = match_for_COICOP_5_code(x, entity, alpha, beta)
        #if match_score > 0:
        #  print(match_score, entity['coicop_code'], entity['root'], entity['terms'] )
        if match_score > max_match_score:
            max_match_score = match_score
            chosen_code = entity['coicop_code']


    if not chosen_code:
        #print(x)
        chosen_code = random.choice([entity['coicop_code'] for entity in entity_forest])
        
    
    return chosen_code


####### EVALUATION #############


def evaluate_model(X_test, y_test, entity_forest, alpha, beta):
    """
    evaluate the performance of the model on a testing set.

    Args:
        X_test (list): list of item descriptions to be classified.
        y_test (list): list of true 5-digit COICOP codes.
        entity_forest (list): list of dictionaries- entity-forest.
        alpha (float): score for root term match.
        beta (float): score for branch term match.

    Returns:
        accuracy of the model on the "test" / classification set.
    """

    correct_predictions = 0
    wrong_predictions = 0
    total_predictions = 0
    
    for i in range(len(X_test)):
        x = X_test.iloc[i]
        y_true = y_test.iloc[i]
        
        # predict the COICOP code using the entity forest and the classify_item_description function
        y_pred = classify_item_description(x, entity_forest, alpha, beta)
        
        if y_pred == y_true:
            correct_predictions += 1
        total_predictions += 1

        if y_pred != y_true:
            #print("------------",'\n', x, y_pred,y_true)
            wrong_predictions += 1

    
    # return accuracy
    accuracy = correct_predictions / total_predictions
    print("total pred: ", total_predictions)
    print("Correct preds:", correct_predictions)
    print("wrond preds", wrong_predictions)
    return accuracy



####### RUN : BUILD + CLASSIFY ##############


from sklearn.model_selection import train_test_split
import time
import os

directory = 'C:/Users/Eier/OneDrive/Documents/Masteroppgave/dataset'
os.chdir(directory)

# read in data
f1 = 'df_conclusive.txt'
df = pd.read_csv(f1,  encoding='iso-8859-1')
df['subclass'] = df['subclass'].astype(str)


f2 = '01_coicop.txt'
coicop = pd.read_csv(f2, encoding='iso-8859-1')
coicop['subclass'] = coicop['subclass'].astype(str)


X = df
y = df['subclass']

start = time.time()

# split the data into building (train) and classification (test) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# build entity forest from 80 percent of data (X_train)
entity_forest = build(X_train, coicop)

# classify and evaluate model on 20 percent (X_test)
X_test_txt = X_test['cleanText']
acc = evaluate_model(X_test_txt, y_test, entity_forest, 10, 2)

end = time.time()
print(f'\nTotal building and evaluation time: {end - start:.2f}')

# calculate the accuracy of the model
print('Accuracy:', acc)
