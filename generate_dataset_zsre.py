import json
import os
from openai import OpenAI
import fire

# Initialize OpenAI API


# Function to extract the relation from the src field
def extract_relation_from_src(src):
    prompt = f"""Examples:
    Question: What sports team was Riki van Steeden a member of?
    Relation: member of
    Question: Who is the president of the United States?
    Relation: president of
    Question: What company was founded by Elon Musk?
    Relation: founded by
    
    Extract the relation from the following question:
    Question: {src}
    
    Provide only the relation in a concise format."""

    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{prompt}"}],
        max_tokens=50
    )
    
    generated_text = completion.choices[0].message.content.strip()
    relation = generated_text
    print(f"Extracted Relation from src: {relation}")

    return relation

# Function to generate a subject-based question
def generate_subject_question(object, relation, original_question):
    # Generate a subject-based question
    prompt = f"""Examples:
    Question: Who is the president of the United States?
    Question: Who founded the company SpaceX?
    Question: Who is a member of the sports team Riki van Steeden?
    
    Original Question: {original_question}
    Generate a new question to ask for the subject based on the object '{object}' and the relation '{relation}'. 
    The generated question must ask for the subject and include both the object '{object}' and relation '{relation}', 
    and follow the format provided in the examples."""
    
    answer = object  # The answer will be the subject in this case (which is expected).
    
    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{prompt}"}],
        max_tokens=100
    )
    
    generated_text = completion.choices[0].message.content.strip()
    
    # Extract the generated question
    question = generated_text.replace("Question:", "").strip()
    print(f"Generated Question: {question}")
    print(f"Generated Answer: {answer}")

    return question, answer

# Function to generate answer aliases for the subject
def generate_answer_aliases(subject):
    example_prompt = """Example:
    Aliases: United States, USA, America"""
    
    strict_instructions = "Please generate synonyms, aliases, or different expressions for the subject using only standard English words, with no special characters or symbols."
    prompt = f"{example_prompt}\n{strict_instructions}\nGenerate aliases for the subject '{subject}'. The output should be in the following format:\nAliases: ..."

    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    generated_text = completion.choices[0].message.content.strip()
    aliases_list = generated_text.replace("Aliases:", "").strip().split(",")
    aliases_list = [alias.strip() for alias in aliases_list]
    
    print(f"Generated Subject Aliases: {aliases_list}")
    return aliases_list

# Function to count already processed items directly from the output file
def count_processed_items(output_filename):
    """Count the number of items already written to the output file."""
    if not os.path.exists(output_filename):
        return 0

    # Open the output file and count the number of JSON objects
    with open(output_filename, 'r') as f:
        content = f.read().strip()

        # Remove the opening and closing brackets to get the content inside
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()

        if not content:
            return 0

        # Split by "}," to count JSON objects (each object ends with `}`)
        return content.count("},") + 1

# Main function to process the dataset with resume capability
def process_subject_type_dataset():
    input_filename = "/data/shared/Alan/LLM_Evaluation/memit_for_mquake/data/zsre_mend_eval.json"
    output_filename = input_filename.replace(".json", "-subject-updated.json")

    # Create the output file if it doesn't exist
    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as f_out:
            f_out.write("[")  # Start the JSON array

    # Count how many items have been processed already
    processed_count = count_processed_items(output_filename)
    print(f"Resuming from item {processed_count}")

    # Load the dataset
    with open(input_filename, 'r') as f:
        data = json.load(f)

    # Resume processing from the next unprocessed item
    for i, case in enumerate(data[processed_count:], start=processed_count):
        subject = case["subject"]
        obj = case["answers"][0]
        src = case["src"]
        print(src)
        # Extract relation from the src question
        relation = extract_relation_from_src(src)

        # Generate new question for subject-based (Who is associated with X and Y?)
        new_question, _ = generate_subject_question(obj, relation, src)
        answer_aliases = generate_answer_aliases(subject)  # Generate aliases for the subject (answer)

        # Create new entry with additional fields (without overwriting existing fields)
        new_entry = {
            "validate_question": new_question,
            "validate_answers": [subject] + answer_aliases  # Add subject with its aliases
        }

        # Write the new entry alongside the original case to the output file
        with open(output_filename, 'a') as f_out:
            if i > 0 or processed_count > 0:
                f_out.write(",\n")  # Add a comma between objects if not the first one
            combined_entry = {**case, **new_entry}  # Merge original and new fields
            json.dump(combined_entry, f_out, indent=4)

    # Close the JSON array if processing is complete
    with open(output_filename, 'a') as f_out:
        f_out.write("\n]")

    print(f"Updated dataset with new subject-based questions and answers saved to {output_filename}.")

# Run the main process
if __name__ == "__main__":
    fire.Fire(process_subject_type_dataset)
