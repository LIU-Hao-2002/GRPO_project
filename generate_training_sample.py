from transformers import AutoTokenizer
from datasets import load_dataset
import pickle
# Load dataset from Hugging Face Hub
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset_id = "./dataset" # 必须是folder
dataset = load_dataset(dataset_id, split="train")
# select a random subset of 5k samples
dataset = dataset.shuffle(seed=42).select(range(5000))

# Load tokenizer from Hugging Face Hub to format the dataset to our "r1" prompt 
tokenizer = AutoTokenizer.from_pretrained("../qwen3b")

# gemerate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(numbers, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
      },
      { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

# convert our dataset to the r1 prompt
dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

with open('dataset/train.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open('dataset/test.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)