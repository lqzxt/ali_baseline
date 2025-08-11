from torch.utils.data import Dataset
import torch
import json

# Custom Dataset class
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, sentence1_str, sentence2_str, task_name='QC'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str
        self.task_name = task_name

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence1 = item[self.sentence1_str]
        sentence2 = item[self.sentence2_str]
        label = item['label']

        if self.task_name == 'QC':
            levels = sentence2.split(',')
            category_lines = '\n'.join([f"- Level {i+1}: {lvl.strip()}" for i, lvl in enumerate(levels)])
            prompt = f"""Does this query belong to this category path? Your answer must be a number (0 or 1)
Answer: (Yes=1 / No=0)


Query: "baking"
Category Path:
- Level 1: food
- Level 2: grocery
- Level 3: flour
- Level 4: baking and cooking
- Level 5: dry ingredients for baking
Answer:1

Query: "10ft sun umbrella"
Category Path:
- Level 1: furniture
- Level 2: outdoor furniture
- Level 3: patio umbrellas & bases
Answer:1

Query: "12mini"
Category Path:
- Level 1: lights & lighting
- Level 2: portable lighting
- Level 3: flashlights & torches
Answer:0

Query: "12v led light"
Category Path:
- Level 1: lights & lighting
- Level 2: professional light
- Level 3: emergency lights
Answer:1

Query: "13 pro max"
Category Path:
- Level 1: computer & office
- Level 2: tablets
Answer:0

Query: "{sentence1}"
Category Path:
{category_lines}
Answer:"""
        else:
            query_part = f'Query: "{sentence1}"'
            category_part = f'Item Title: "{sentence2}"'
            qa_prompt = 'Question: Does this query belong to this item title?\n\nAnswer (Yes=1 / No=0):'
            prompt = query_part + '\n' + category_part + '\n' + qa_prompt
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(label), dtype=torch.long)
        }

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        return self.data[idx]







# from torch.utils.data import Dataset
# import torch
# import json

# # Custom Dataset class
# class SentencePairDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length, sentence1_str, sentence2_str):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data = []
#         self.sentence1_str = sentence1_str
#         self.sentence2_str = sentence2_str

#         with open(file_path, 'r') as f:
#             for line in f:
#                 item = json.loads(line.strip())
#                 self.data.append(item)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         sentence1 = item[self.sentence1_str]
#         sentence2 = item[self.sentence2_str]
#         label = item['label']

#         encoding = self.tokenizer(
#             sentence1,
#             sentence2,
#             max_length=self.max_length,
#             padding=False,
#             truncation=True,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(int(label), dtype=torch.long)
#         }

# class SentencePairPredictDataset(SentencePairDataset):
#     def __getitem__(self, idx):
#         return self.data[idx]