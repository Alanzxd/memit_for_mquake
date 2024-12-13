import json
from openai import OpenAI

# Initialize OpenAI API


def generate_question_and_cloze(triple, prompt_type, original_question, original_cloze):
    subject, relation, obj = triple

    if prompt_type == 0:  # Generate question and cloze for the relationship
        prompt = f"""Examples:
        Question: What is the relationship between Ellie Kemper and Croatia?
        Cloze: The relationship between Ellie Kemper and Croatia is
        Question: How is Barack Obama related to the United States of America?
        Cloze: Barack Obama is related to the United States of America as
        Question: What is the relationship between Elon Musk and SpaceX?
        Cloze: The relationship between Elon Musk and SpaceX is
        
        Generate a new question and a cloze using the subject '{subject}' and object '{obj}' to ask for the relation. The cloze must be an incomplete sentence where the answer is blanked out, strictly following the format provided in the examples. Ensure that the relation does not appear in the question or cloze. Do not use underscores, dashes, or other special characters in the cloze. The generated question and cloze must only include both the subject '{subject}' and object '{obj}'.
        
        Rule: 
        1. Please strictly follow the format provided in the examples.
        2. Do not introduce new information or modify the intended meaning.
        3. The generated question and cloze should be clear and free of any ambiguity. For example, instead of writing 'capital of Berlin', please phrase it as 'the country which capital is Berlin'.
        4. Consider the question and cloze separately.
        5. Any qualifiers or descriptors present in the question must also be reflected in the cloze. For example, if the question is 'Who is the head of government that represents the country led by Justin Trudeau?', the cloze should explicitly mention 'The head of government of the country led by Justin Trudeau is' rather than just 'The head of government is'.
        6. The cloze must be an incomplete sentence without any punctuation marks, and should not include words like 'not' or other modifiers that reverse or alter the meaning."""
        
        answer = relation  # Use relation as the answer
    
    elif prompt_type == 1:  # Generate question and cloze for the subject
        prompt = f"""Examples:
        Question: Who is the head of state of Croatia?
        Cloze: The head of state of Croatia is
        Question: Who was the founder of Apple Inc.?
        Cloze: The founder of Apple Inc. is
        Question: Who leads the United Nations?
        Cloze: The leader of the United Nations is
        
        Original Question: {original_question}
        Original Cloze: {original_cloze}
        Original Triple: {triple}

        Generate a new question and a cloze based on the original question and cloze provided above. Please use the object '{obj}' and relation '{relation}' to ask for the subject '{subject}'. The cloze must be an incomplete sentence where the answer is blanked out, strictly following the format provided in the examples. Ensure that the subject '{subject}' does not appear in the question or cloze. Do not use underscores, dashes, or other special characters in the cloze. The generated question and cloze must only include both the object '{obj}' and relation '{relation}'.
        
        Rule: 
        1. Please strictly follow the format provided in the examples.
        2. Do not introduce new information or modify the intended meaning.
        3. The generated question and cloze should be clear and free of any ambiguity. For example, instead of writing 'capital of Berlin', please phrase it as 'the country which capital is Berlin'.
        4. Consider the question and cloze separately.
        5. Any qualifiers or descriptors present in the question must also be reflected in the cloze. For example, if the question is 'Who is the head of government that represents the country led by Justin Trudeau?', the cloze should explicitly mention 'The head of government of the country led by Justin Trudeau is' rather than just 'The head of government is'.
        6. The cloze must be an incomplete sentence without any punctuation marks, and should not include words like 'not' or other modifiers that reverse or alter the meaning."""
        answer = subject  # Use subject as the answer
    
    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=200
    )

    generated_text = completion.choices[0].message.content.strip()
    
    # Separate the generated text into question and cloze
    question_part, cloze_part = generated_text.split("Cloze:", 1)

    question = question_part.replace("Question:", "").strip()
    cloze = cloze_part.strip()
    print(question)
    print(cloze)
    print(answer)
    return question, cloze, answer

def generate_answer_aliases(answer):
    example_prompt = """Example:
    Aliases: United States, USA, America"""
    
    strict_instructions = "Please generate synonyms or different expressions with the same meaning using only standard English words, with no special characters or symbols."

    prompt = f"{example_prompt}{strict_instructions}\nGenerate aliases for the answer '{answer}'. The output should be in the following format:\nAliases: ..."

    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    generated_text = completion.choices[0].message.content.strip()

    # Extract and split the generated alias list
    aliases_list = generated_text.replace("Aliases:", "").strip().split(",")

    aliases_list = [alias.strip() for alias in aliases_list]
    return aliases_list

# Load the dataset from the file
with open('/data/shared/Alan/LLM_Evaluation/memit_for_mquake/data/MQuAKE-hard.json', 'r') as f:
    data = json.load(f)

# 初始化 prompt_type 计数器
prompt_counter = 0

# 定义需要跳过的关键词列表
skip_keywords = [
    "citizenship", "origin", "death", "educated", "location", 
    "employer", "continent", "position", "birth", "sport", 
    "genre", "occupation"
]

# Process each case
for case in data:
    new_triples = case["orig"]["new_triples_labeled"]
    original_single_hops = case["new_single_hops"]
    new_single_hops = []

    for i, triple in enumerate(new_triples):
        prompt_type = prompt_counter % 2  # 只在 type0 和 type1 之间切换
        
        # 获取原始问题和 cloze
        original_hop = original_single_hops[i]
        original_question = original_hop["question"]
        original_cloze = original_hop["cloze"]
        
        # 如果是type1，并且relation中包含指定关键词，跳过当前循环
        if prompt_type == 1 and any(keyword in triple[1].lower() for keyword in skip_keywords):
            print(f"skipping due to keyword in relation: {triple[1]}")
            # 不增加 prompt_counter，保持下一次循环仍然为 type1
            continue
        
        # 生成新的问题和 cloze
        question, cloze, answer = generate_question_and_cloze(triple, prompt_type, original_question, original_cloze)
        answer_aliases = generate_answer_aliases(answer)
        new_single_hops.append({
            "question": question,
            "cloze": cloze,
            "answer": answer,
            "answer_alias": answer_aliases
        })

        # 正常完成循环后增加计数器
        prompt_counter += 1
    
    # 将结果添加到 case 中
    case["new_single_hops"] = new_single_hops

# 保存结果到文件
with open("updated_cases_hard.json", "w") as f:
    json.dump(data, f, indent=4)

print(prompt_counter)
print("New single hop questions, clozes, answers, and answer aliases generated and saved.")
