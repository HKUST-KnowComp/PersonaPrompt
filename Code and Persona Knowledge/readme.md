Code For PresonaPrompt

It contains the following file:
* PresonaPrompt.py
* PresonaVerbalizer.py
* data_utils.py
* requirements.txt
* PresonaKnowledge4dimension_trainset_processed.json
* PresonaKnowledge4dimension_devset_processed.json
* PresonaKnowledge4dimension_testset_processed.json

## Prerequisites
* All Environment requirements list in "requirements.txt"

## Usage
1. You should download Kailo dataset and DDO dataset from "https://esdurmus.github.io/".
2. Run Command: "pip install -r requirements.txt".
3. Run following command:
"python PresonaPrompt.py
--setting argu-PersonaKnowledge
--template_id 0 
--max_steps 30000 
--batch_size 4
--batch_size_e 4
--max_seq_length 512 
--eval_every_steps 250
--model_name_or_path google/flan-t5-base
--result_file ../PersonaKnowledge_7e-7.txt
--ckpt_file PersonaKnowledge_7e-7
--prompt_lr 0.0000007"