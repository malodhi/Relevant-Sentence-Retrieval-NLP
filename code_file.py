# -*- coding: utf-8 -*-

"""

Created on Thu Oct 22 12:27:06 2020
@author: mansoor.lodhi

"""

from os import path
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from abydos.distance import dist_levenshtein as lev_dist
from sentence_transformers import SentenceTransformer

stop_words =  set(stopwords.words('english'))
tokenizer  = nltk.RegexpTokenizer(r"\w+") 


class Similarity_Model:
        
    def __init__(self, file_path = ""):
        
        self.file_path = file_path
        self.data = None    
        self.embedded_data = None
        self.preprocessed_data = None
        
        self.read_data()
        
        # Using NLTK library lets remove puntuations and stop_words from stored observations
        self.preprocessed_data = [self.preprocess_query(observation) for observation in self.data]
        print("Stored Observations Preprocessed.")
        
        # Below we load the model for strings encoding        
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        print("Model Uploaded Successfully.")
                
        # Below we embedd the read data to use it at runtime
        self.embedded_data = self.model.encode(self.data)
        print("Provided Observation Details Embedded.")
        
        
        
    def read_data(self):
        
        # Read the excel file into dataframe and remove any redundant data.
        data_df = pd.read_excel(self.file_path).drop_duplicates()
        
        # Raise Exception if we don't have any data to compare the user observation.
        if data_df.empty:
            raise Exception('The Input Data File is Empty !')
            
        print("Total Unique Stored Observations :  ", data_df.shape[0])
        
        data_lists = data_df.values.tolist()
        
        self.data = [ls[0] for ls in data_lists if ls[0]] 
    
    def preprocess_query(self, query):
        """
        query --> string
        """
        
        list_of_words = tokenizer.tokenize(query)

        filtered_list_of_words = [word.strip().replace("\r", " ").replace("\n", " ").lower() 
                                  for word in list_of_words if word not in stop_words]

        preprocessed_query_string = " ".join(filtered_list_of_words)
        
        return preprocessed_query_string
    
    def cosine(self, u, v):
        """
        u shape -> (1, feature_vector)
        v shape -> (stored_observations, feature_vector)
        output shape  -> (stored_observations, 1)
        """
        
        u = u.reshape(1,-1).T
        assert u.shape[0] == v.shape[1], "Query Embedding doesn't match Built in Embedding (Vector)"
        output_vecor = np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v))
        
        return output_vecor

    def use_keyword_search_algo(self, processed_query, suggested_observations=5):
        
        matched_keywords_to_string_len = []
        
        for observation in self.preprocessed_data:

            string_len = len(observation.split())
            number_of_keywords_in_query_matched  = 0

            for keyword in processed_query.split():

                if keyword in observation.split():

                    number_of_keywords_in_query_matched += 1
            
            matched_keywords_to_string_len.append(number_of_keywords_in_query_matched / string_len)
        
        highest_similarity_indices = np.argsort(matched_keywords_to_string_len)[::-1]
        most_relevant_observations = []
        
        # In Below loop we retrieve most relevant strings based on the above obtained indices.
        for i in range(suggested_observations):
            most_relevant_observations.append(self.data[highest_similarity_indices[i]])
    
        return most_relevant_observations
        
    def use_distance_seach_algo(self, processed_query, suggested_observations = 5):
        
        ratio_of_observation_matched = []        
        
        for observation in self.preprocessed_data:            
            ratio_of_observation_matched.append(1 - lev_dist(processed_query, observation, mode='osa'))

        highest_similarity_indices = np.argsort(ratio_of_observation_matched)[::-1]
        most_relevant_observations = []
        
        # In Below loop we retrieve most relevant strings based on the above obtained indices.
        for i in range(suggested_observations):
            if ratio_of_observation_matched[i] > 0.2:
                most_relevant_observations.append(self.data[highest_similarity_indices[i]])

        return most_relevant_observations
        
    def use_bert_model(self, query, already_matched_observations , suggested_observations):
        
        embedded_query = self.model.encode(query)
        
        # The Below Result Vector Will be the cosine similarity between query and each string in data.
        similarity_result_vector = self.cosine(embedded_query , self.embedded_data)
        assert similarity_result_vector.shape[0] == self.embedded_data.shape[0]
                                          
        similarity_vector = [vect[0] for vect in similarity_result_vector]
        
        # Below we obtain the indices of strings which have highest similarity measure.
        highest_similarity_indices = np.argsort(similarity_vector)[::-1]
            
        most_relevant_observations = []
        
        # In Below loop we retrieve most relevant strings based on the above obtained indices.
        for i in range(suggested_observations):
            observation = self.data[highest_similarity_indices[i]]
            if observation not in already_matched_observations:
                most_relevant_observations.append(observation)
        
        return most_relevant_observations
        
    def find_top_matched_observations(self, query, suggested_observations = 5):
        
        assert isinstance(query, str), "Input Data Not a String."
        if suggested_observations > len(self.data) : suggested_observations = len(self.data)

        preprocessed_query = self.preprocess_query(query)
        preprocessed_query_len = len(preprocessed_query.split()) if preprocessed_query else 0 
        
        if preprocessed_query_len  == 0:
            raise Exception("Please enter a valid string.") 
        
        if preprocessed_query_len < 6:
            top_matched_observations = self.use_keyword_search_algo(preprocessed_query, suggested_observations)
    
        elif preprocessed_query_len >= 6:
            
            top_matched_observations  = self.use_distance_seach_algo(preprocessed_query)
            no_observations_found = len(top_matched_observations)
            more_top_matched_observations = []
            if suggested_observations-no_observations_found != 0:
                more_observations_required = suggested_observations - no_observations_found

                # We have to make sure that bert_model doesn't produce the similar result as that obtained  using distance method.
                more_top_matched_observations = self.use_bert_model(query, top_matched_observations, suggested_observations=
                                                                    more_observations_required)


            top_matched_observations += more_top_matched_observations
            
        return top_matched_observations


        


if __name__=="__main__":    
    ### Below the model is uploaded once and for all. The stored observations are embedded
    ### only once at the very beginning to save time and resrouces at runtime.
    handler_obj = Similarity_Model(file_path="D:\dev\SORs.xlsx")
    
    query = ""
    
    while query != "exit":
        
        query = input("Enter Your Query : ")
        if query == "exit" : continue
    
        suggested_observations =  input("Desired Suggestions : ")
        try:
            suggested_observations = int(suggested_observations)
            if suggested_observations <= 0 : raise Exception("")
        except:
            print("Enter a Proper Suggestion Value")
            continue
        
        
    
        try:
            relevant_searches = handler_obj.find_top_matched_observations(query, suggested_observations=suggested_observations)
        except Exception as e:
            print("Error :  " ,e)
            continue
        
        print("Relevant Searches : ", end="\n")
        
        for i in range(suggested_observations):
            print(f"{i+1}. {relevant_searches[i]} ")
        

        
        
        