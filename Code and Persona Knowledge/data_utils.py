from openprompt.data_utils import PROCESSORS
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from typing import List, Dict, Callable, Sequence
from tqdm import tqdm
import torch
import numpy as np
import time
import os
import re
import json
import random
from random import sample

class KialoProcessor(DataProcessor):
    
    discourse_labels_mapping = {
    "<Null>" : "null",  
    "<Concession>": "nonetheless", #"if", 
    "<Contrast>": "however", 
    "<Reason>": "because", 
    "<Result>": "so", 
    "<Condition>":"if",
    "<Alternative>":"unless", 
    "<ChosenAlternative>":"instead", 
    "<Conjunction>":"also", 
    "<Exception>":"except", 
    "<Instantiation>": "for", #"for example",
    "<Restatement>":"specifically",
    "<Precedence>":"before",
    "<Succession>":"after",  
    "<Synchrony>":"when",
    }
    
    discourse_relation_mapping = {
    "<Null>" : "null",  
    "<Concession>": "Concession", #"if", 
    "<Contrast>": "Contrast", 
    "<Reason>": "Reason", 
    "<Result>": "Result", 
    "<Condition>":"Condition",
    "<Alternative>":"Alternative", 
    "<ChosenAlternative>":"ChosenAlternative", 
    "<Conjunction>":"Conjunction", 
    "<Exception>":"Exception", 
    "<Instantiation>": "Instantiation", #"for example",
    "<Restatement>":"Restatement",
    "<Precedence>":"Precedence",
    "<Succession>":"Succession",  
    "<Synchrony>":"Synchrony",
    }
    
    
    stance_mapping = {
    "<null>" : "null",
    "<pro>" : "Support",
    "<con>" : "Oppose",
    }
    
    def __init__(self):
        super().__init__()
        self.labels = ["IMPACTFUL", "MEDIUM IMPACT", "NOT IMPACTFUL"]
        
    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f.readlines():
                example_json = json.loads(line)
                labels = example_json["label"]
                label = self.get_label_id(labels)
                
                guid = example_json['id']
                context = " ".join(example_json['context'])
                context_l3 = " ".join(example_json['context'][-3:])
                if len(context.split()) > 150:
                    context = " ".join(context.split()[-150:])
                if len(context_l3.split()) > 150:
                    context_l3 = " ".join(context_l3.split()[-150:])
                meta = {
                    'context': context,
                    'context_l2': " ".join(example_json['context'][-2:]),
                    'context_l3': context_l3,
                    'argument1': example_json["context"][-1],
                    'argument2': " ".join(example_json["text"]),
                    'discourse_label': self.discourse_relation_mapping[example_json["discourse_label"][-1]],
                    'stance': self.stance_mapping[example_json["stance_label"][-1]],
                    'conncetive_label': self.discourse_labels_mapping[example_json["discourse_label"][-1]],
                }                
                example = InputExample(guid=guid, label=label, meta=meta)
                examples.append(example)
        return examples

class DDOProcessor(DataProcessor):

#     stance_mapping = {
#     "<null>" : "null",
#     "<pro>" : "Support",
#     "<con>" : "Oppose",
#     }

    label_map = {
        "FIRST"  : "First Speaker",
        "SECOND" : "Second Speaker",
    }
  
    def __init__(self):
        super().__init__()
        self.labels = ["First Speaker", "Second Speaker"]
        
    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for line in f.readlines():
                example_json = json.loads(line)
                labels = self.label_map[example_json["label"].split(" ")[0]]
                label = self.get_label_id(labels)
                
                guid = example_json['id']
                argument2 = " ".join(example_json["text"]).replace("Affirmative","\n\nPro").replace("Negative", "\n\nCon")
                context = " ".join(example_json['context'])#.replace("\n","")
#                 context = " ".join(example_json['context']).replace("Affirmative","\n\nAffirmative").replace("Negative", "\n\nNegative")
#                 context = " ".join(example_json['context']) #+ " ".join(example_json['text'])  #example_json['context'][0] + example_json['context'][1] + example_json['context'][2]
#                 context = example_json['context'][0] + example_json['context'][1] + example_json['context'][2] + example_json['context'][3] + example_json['context'][4] + " ".join(example_json['text'])  #
#                 context_l3 = " ".join(example_json['context'][-3:])
#                 if len(context.split()) > 150:
#                     context = " ".join(context.split()[-150:])
#                 if len(context_l3.split()) > 150:
#                     context_l3 = " ".join(context_l3.split()[-150:])
                meta = {
                    'context': context,
#                     'context_l2': " ".join(example_json['context'][-2:]),
#                     'context_l3': context_l3,
#                     'argument1': example_json["context"][-1],
                    'argument2': argument2,
#                     'discourse_label': self.discourse_relation_mapping[example_json["discourse_label"][-1]],
#                     'stance': self.stance_mapping[example_json["stance_label"][-1]],
#                     'conncetive_label': self.discourse_labels_mapping[example_json["discourse_label"][-1]],
                }                
                example = InputExample(guid=guid, label=label, meta=meta)
                examples.append(example)
        return examples

class PersonaKnowledge_DDOProcessor(DataProcessor):
    
    discourse_labels_mapping = {
    "<Null>" : "null",  
    "<Concession>": "nonetheless", #"if", 
    "<Contrast>": "however", 
    "<Reason>": "because", 
    "<Result>": "so", 
    "<Condition>":"if",
    "<Alternative>":"unless", 
    "<ChosenAlternative>":"instead", 
    "<Conjunction>":"also", 
    "<Exception>":"except", 
    "<Instantiation>": "for", #"for example",
    "<Restatement>":"specifically",
    "<Precedence>":"before",
    "<Succession>":"after",  
    "<Synchrony>":"when",
    }
    
    discourse_relation_mapping = {
    "<Null>" : "null",  
    "<Concession>": "Concession", #"if", 
    "<Contrast>": "Contrast", 
    "<Reason>": "Reason", 
    "<Result>": "Result", 
    "<Condition>":"Condition",
    "<Alternative>":"Alternative", 
    "<ChosenAlternative>":"ChosenAlternative", 
    "<Conjunction>":"Conjunction", 
    "<Exception>":"Exception", 
    "<Instantiation>": "Instantiation", #"for example",
    "<Restatement>":"Restatement",
    "<Precedence>":"Precedence",
    "<Succession>":"Succession",  
    "<Synchrony>":"Synchrony",
    }
    
#     stance_mapping = {
#     "<null>" : "null",
#     "<pro>" : "Support",
#     "<con>" : "Oppose",
#     }
    
    stance_mapping = {
    "<null>" : "null; ",
    "<pro>" : "Pro; ",
    "<con>" : "Con; ",
    }
    
    number_mapping = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.']
    
    label_map = {
        "FIRST"  : "First Speaker",
        "SECOND" : "Second Speaker",
    }
  
    def __init__(self):
        super().__init__()
        self.labels = ["First Speaker", "Second Speaker"]
    
    #kownledge prompt
    def read_files(self, path):
        ground_knowledge=[]
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                ground_knowledge.append(example_json)
        return ground_knowledge

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        #Get Ground Knowledge
        if split == "train2020":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_trainset_DDO.json"
        elif split == "valid2020":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_validset_DDO.json" 
        elif split == "test2020":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_testset_DDO.json"      
        ground_knowledge = self.read_files(knowledge_path)
            
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                labels = self.label_map[example_json["label"].split(" ")[0]]
                label = self.get_label_id(labels)
                
                guid = example_json['id']
                context = " ".join(example_json['context'])
#                 context_l3 = " ".join(example_json['context'][-3:])
#                 if len(context.split()) > 150: #250
#                     context = " ".join(context.split()[-150:])
#                 if len(context_l3.split()) > 150:
#                     context_l3 = " ".join(context_l3.split()[-150:])
                
                PersonaKnowledge = ground_knowledge[0][choicex]
                Processed_PersonaKnowledge = ""
                for i in range(len(PersonaKnowledge)):
                    Processed_PersonaKnowledge = Processed_PersonaKnowledge + " Role: " + PersonaKnowledge[i][0]+ "; "  + PersonaKnowledge[i][1]+ " " + PersonaKnowledge[i][2]+ " " + PersonaKnowledge[i][3]+ " " + PersonaKnowledge[i][4]
                Processed_PersonaKnowledge = Processed_PersonaKnowledge.strip(' ')

#                 stance_text = ""
#                 for instance_stance in example_json["stance_label"]:
#                     stance_text = stance_text + self.stance_mapping[instance_stance]
                
#                 context_stance_text = ""
#                 for context_i in range(len(example_json["context"])):
#                     context_stance_text = context_stance_text + example_json['context'][context_i] + self.stance_mapping[example_json["stance_label"][context_i + 1]]
#                 if len(context_stance_text.strip(' ').split()) > 150:
#                     context_stance_text = " ".join(context_stance_text.strip(' ').split()[-150:])

                meta = {
                    'context': context,
#                     'context_l3': context_l3,
#                     'context_l2': " ".join(example_json['context'][-2:]),
#                     'context_l1': example_json["context"][-1],
                    'argument2': " ".join(example_json["text"]),
#                     'discourse_label': self.discourse_relation_mapping[example_json["discourse_label"][-1]],
#                     'context_stance': context_stance_text,                    
#                     'stance': stance_text.strip(' '),
#                     'stance': self.stance_mapping[example_json["stance_label"][-1]],                    
#                     'conncetive_label': self.discourse_labels_mapping[example_json["discourse_label"][-1]],
#                     'Persona_Knowledge': "".join(ground_knowledge[0][choicex]['choices'][0]['message']['content']),
                    'Persona_Knowledge': Processed_PersonaKnowledge,  
                }                
                example = InputExample(guid=guid, label=label, meta=meta)
                if choicex == 0:
                    print(example)
                examples.append(example)
        return examples

class PersonaKnowledgeProcessor(DataProcessor):
    
    discourse_labels_mapping = {
    "<Null>" : "null",  
    "<Concession>": "nonetheless", #"if", 
    "<Contrast>": "however", 
    "<Reason>": "because", 
    "<Result>": "so", 
    "<Condition>":"if",
    "<Alternative>":"unless", 
    "<ChosenAlternative>":"instead", 
    "<Conjunction>":"also", 
    "<Exception>":"except", 
    "<Instantiation>": "for", #"for example",
    "<Restatement>":"specifically",
    "<Precedence>":"before",
    "<Succession>":"after",  
    "<Synchrony>":"when",
    }
    
    discourse_relation_mapping = {
    "<Null>" : "null",  
    "<Concession>": "Concession", #"if", 
    "<Contrast>": "Contrast", 
    "<Reason>": "Reason", 
    "<Result>": "Result", 
    "<Condition>":"Condition",
    "<Alternative>":"Alternative", 
    "<ChosenAlternative>":"ChosenAlternative", 
    "<Conjunction>":"Conjunction", 
    "<Exception>":"Exception", 
    "<Instantiation>": "Instantiation", #"for example",
    "<Restatement>":"Restatement",
    "<Precedence>":"Precedence",
    "<Succession>":"Succession",  
    "<Synchrony>":"Synchrony",
    }
    
#     stance_mapping = {
#     "<null>" : "null",
#     "<pro>" : "Support",
#     "<con>" : "Oppose",
#     }
    
    stance_mapping = {
    "<null>" : "null; ",
    "<pro>" : "Pro; ",
    "<con>" : "Con; ",
    }
    
    number_mapping = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.']
    
    def __init__(self):
        super().__init__()
        self.labels = ["IMPACTFUL", "MEDIUM IMPACT", "NOT IMPACTFUL"]
    
    #kownledge prompt
    def read_files(self, path):
        ground_knowledge=[]
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                ground_knowledge.append(example_json)
        return ground_knowledge

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        #Get Ground Knowledge
        if split == "train":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_trainset_processed.json"
        elif split == "dev":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_devset_processed.json" 
        elif split == "test":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_testset_processed.json"      
        ground_knowledge = self.read_files(knowledge_path)
            
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                labels = example_json["label"]
                label = self.get_label_id(labels)
                
                guid = example_json['id']
                context = " ".join(example_json['context'])
                context_l3 = " ".join(example_json['context'][-3:])
                if len(context.split()) > 150: #250
                    context = " ".join(context.split()[-150:])
                if len(context_l3.split()) > 150:
                    context_l3 = " ".join(context_l3.split()[-150:])
                
                PersonaKnowledge = ground_knowledge[0][choicex]
                Processed_PersonaKnowledge = ""
                for i in range(len(PersonaKnowledge)):
                    Processed_PersonaKnowledge = Processed_PersonaKnowledge + " Role: " + PersonaKnowledge[i][0]+ "; "  + PersonaKnowledge[i][1]+ " " + PersonaKnowledge[i][2]+ " " + PersonaKnowledge[i][3]+ " " + PersonaKnowledge[i][4]
                Processed_PersonaKnowledge = Processed_PersonaKnowledge.strip(' ')

                stance_text = ""
                for instance_stance in example_json["stance_label"]:
                    stance_text = stance_text + self.stance_mapping[instance_stance]
                
                context_stance_text = ""
                for context_i in range(len(example_json["context"])):
                    context_stance_text = context_stance_text + example_json['context'][context_i] + self.stance_mapping[example_json["stance_label"][context_i + 1]]
#                 if len(context_stance_text.strip(' ').split()) > 150:
#                     context_stance_text = " ".join(context_stance_text.strip(' ').split()[-150:])

                meta = {
                    'context': context,
                    'context_l3': context_l3,
                    'context_l2': " ".join(example_json['context'][-2:]),
                    'context_l1': example_json["context"][-1],
                    'argument2': " ".join(example_json["text"]),
                    'discourse_label': self.discourse_relation_mapping[example_json["discourse_label"][-1]],
                    'context_stance': context_stance_text,                    
                    'stance': stance_text.strip(' '),
#                     'stance': self.stance_mapping[example_json["stance_label"][-1]],                    
                    'conncetive_label': self.discourse_labels_mapping[example_json["discourse_label"][-1]],
#                     'Persona_Knowledge': "".join(ground_knowledge[0][choicex]['choices'][0]['message']['content']),
                    'Persona_Knowledge': Processed_PersonaKnowledge,  
                }                
                example = InputExample(guid=guid, label=label, meta=meta)
                if choicex == 0:
                    print(example)
                examples.append(example)
        return examples

class PersonaKnowledge_ablationstudy(DataProcessor):
    
    discourse_labels_mapping = {
    "<Null>" : "null",  
    "<Concession>": "nonetheless", #"if", 
    "<Contrast>": "however", 
    "<Reason>": "because", 
    "<Result>": "so", 
    "<Condition>":"if",
    "<Alternative>":"unless", 
    "<ChosenAlternative>":"instead", 
    "<Conjunction>":"also", 
    "<Exception>":"except", 
    "<Instantiation>": "for", #"for example",
    "<Restatement>":"specifically",
    "<Precedence>":"before",
    "<Succession>":"after",  
    "<Synchrony>":"when",
    }
    
    discourse_relation_mapping = {
    "<Null>" : "null",  
    "<Concession>": "Concession", #"if", 
    "<Contrast>": "Contrast", 
    "<Reason>": "Reason", 
    "<Result>": "Result", 
    "<Condition>":"Condition",
    "<Alternative>":"Alternative", 
    "<ChosenAlternative>":"ChosenAlternative", 
    "<Conjunction>":"Conjunction", 
    "<Exception>":"Exception", 
    "<Instantiation>": "Instantiation", #"for example",
    "<Restatement>":"Restatement",
    "<Precedence>":"Precedence",
    "<Succession>":"Succession",  
    "<Synchrony>":"Synchrony",
    }
    
    stance_mapping = {
    "<null>" : "null",
    "<pro>" : "Support",
    "<con>" : "Oppose",
    }
    
    number_mapping = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.']
    
    def __init__(self):
        super().__init__()
        self.labels = ["IMPACTFUL", "MEDIUM IMPACT", "NOT IMPACTFUL"]
    
    #kownledge prompt
    def read_files(self, path):
        ground_knowledge=[]
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                ground_knowledge.append(example_json)
        return ground_knowledge

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        #Get Ground Knowledge
        if split == "train":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_trainset_processed.json"
        elif split == "dev":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_devset_processed.json" 
        elif split == "test":
            knowledge_path = "./PromptArgument/PresonaKnowledge4dimension_testset_processed.json"      
        ground_knowledge = self.read_files(knowledge_path)
            
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                labels = example_json["label"]
                label = self.get_label_id(labels)
                guid = example_json['id']
                context = " ".join(example_json['context'])
                if len(context.split()) > 150: #250
                    context = " ".join(context.split()[-150:])
                
                PersonaKnowledge = ground_knowledge[0][choicex]
                Processed_PersonaKnowledge = ""
                for i in range(len(PersonaKnowledge)):
                    if i == 4: #knowledge number
                        break
                    Processed_PersonaKnowledge = Processed_PersonaKnowledge + " Role: " + PersonaKnowledge[i][0]+ "; "  + PersonaKnowledge[i][1]+ " " + PersonaKnowledge[i][2]+ " " + PersonaKnowledge[i][3]+ " " + PersonaKnowledge[i][4]
                Processed_PersonaKnowledge = Processed_PersonaKnowledge.strip(' ')

                meta = {
                    'context': context,
                    'context_l2': " ".join(example_json['context'][-2:]),
                    'argument1': example_json["context"][-1],
                    'argument2': " ".join(example_json["text"]),
                    'discourse_label': self.discourse_relation_mapping[example_json["discourse_label"][-1]],
                    'stance': self.stance_mapping[example_json["stance_label"][-1]],
                    'conncetive_label': self.discourse_labels_mapping[example_json["discourse_label"][-1]],
#                     'Persona_Knowledge': "".join(ground_knowledge[0][choicex]['choices'][0]['message']['content']),
                    'Persona_Knowledge': Processed_PersonaKnowledge,  
                }                
                example = InputExample(guid=guid, label=label, meta=meta)
                if choicex == 0:
                    print(example)
                examples.append(example)
        return examples
