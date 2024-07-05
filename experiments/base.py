import json
import typing
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
import unicodedata

# 数据集类
class MQuAKE_T(Dataset):
    """
    Dataset class for loading MQuAKE-T data.
    """
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_loc = data_dir / "MQuAKE-T.json"
        if not mquake_loc.exists():
            remote_url = f"{REMOTE_ROOT}/MQuAKE-T.json"
            print(f"{mquake_loc} does not exist. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, mquake_loc)
        
        with open(mquake_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded MQuAKE-T dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

# Helper functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    multi_hop_prompt: str,
    edit_prompts: dict
) -> typing.Dict:
    multi_hop_accuracy, multi_hop_answers = calculate_multi_hop_accuracy(
        model, tokenizer, record, multi_hop_prompt
    )
    edit_success_rate, instance_accuracy, edit_answers = calculate_edit_accuracy(
        model, tokenizer, record, edit_prompts
    )

    generated_answers = multi_hop_answers + edit_answers

    print(f"Multi-hop Accuracy: {multi_hop_accuracy}")
    print(f"Edit-wise Success Rate: {edit_success_rate}")
    print(f"Instance-wise Accuracy: {instance_accuracy}")

    return {
        'multi_hop_accuracy': multi_hop_accuracy,
        'edit_success_rate': edit_success_rate,
        'instance_accuracy': instance_accuracy,
        'questions': record['questions'],
        'original_answers': [rw['target_true']['str'] for rw in record['requested_rewrite']],
        'generated_answers': generated_answers,
    }

def calculate_multi_hop_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    multi_hop_prompt: str
):
    correct_responses = 0
    generated_answers = []
    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    extended_answers = record.get('answer_extended', [])

    for question in questions:
        full_prompt = multi_hop_prompt + "\n" + question
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = generate_fast(
            model, tokenizer, inputs["input_ids"], top_k=5, max_length=100
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_answers.append(generated_text)

        print(f"Question: {question}")
        print(f"Generated Answer: {generated_text}")

        if correct_answer.lower() in generated_text.lower() or any(alias.lower() in generated_text.lower() for alias in answer_aliases) or any(answer.lower() in generated_text.lower() for answer in extended_answers):
            correct_responses += 1

    multi_hop_accuracy = correct_responses / len(questions)
    return multi_hop_accuracy, generated_answers

def calculate_edit_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    edit_prompts: dict
):
    success_count = 0
    all_facts_recalled = True
    generated_answers = []
    requested_rewrite = record['requested_rewrite']

    for rewrite in requested_rewrite:
        question = rewrite['prompt'].format(rewrite['subject'])
        prompt_key = rewrite['prompt_key']
        full_prompt = edit_prompts[prompt_key] + "\n" + question
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = generate_fast(
            model, tokenizer, inputs["input_ids"], top_k=5, max_length=100
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_answers.append(generated_text)

        print(f"Rewrite Question: {question}")
        print(f"Generated Answer: {generated_text}")

        target_new = rewrite['target_new']['str']

        if target_new.lower() in generated_text.lower():
            success_count += 1

    all_facts_recalled = (success_count == len(requested_rewrite))
    edit_success_rate = success_count / len(requested_rewrite)
    instance_accuracy = 1 if all_facts_recalled else 0

    return edit_success_rate, instance_accuracy, generated_answers

def generate_fast(
    model, tokenizer, input_ids, top_k=5, max_length=100
):
    model.eval()
    cur_context = slice(input_ids.size(1))
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    batch_size = input_ids.size(0)
    max_out_len = input_ids.size(1) + max_length

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:
            model_out = model(input_ids, attention_mask=attention_mask, use_cache=True)
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tokenizer.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("", "")
        for x in txt
    ]

    return txt

def main(
    model_name: str,
    ds_name: str,
    dataset_size_limit: int,
    generation_test_interval: int,
    dir_name: str,
    multi_hop_prompt: str,
    edit_prompts: dict
):
    current_dir = Path.cwd()
    results_dir = current_dir / "results" / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {results_dir}")

    print("Instantiating model")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    print("Loading dataset")
    data_dir = Path("data")
    dataset = MQuAKE_T(data_dir, size=dataset_size_limit)

    new_results_dir = results_dir / "evaluation" / "original_model"
    new_results_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating all data with the original model...")
    for record_chunks in chunks(dataset, 1):
        for record in record_chunks:
            out_file = Path(new_results_dir / f"original_model_case_{record['case_id']}.json")
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            
            start = time()
            exec_time = time() - start
            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": [record["case_id"]],
                "num_edits": 0,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": compute_rewrite_quality_mquake(
                    model,
                    tok,
                    record,
                    multi_hop_prompt,
                    edit_prompts
                ),
            }

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
            print(f"Evaluation took {time() - start} seconds")

if __name__ == "__main__":
    multi_hop_prompt = """Q: What is the country where The Rotunda is located? A: United States of America
Q: In which country was Tohar Butbul granted citizenship? A: Israel
Q: Who was Nissan 200SX created by? A: Nissan
Q: What continent is the country where Prickly Pear grows located in? A: Europe
Q: What is the capital of the country where Plainfield Town Hall is located? A: Washington, D.C.
Q: In which country is the company that created Nissan 200SX located? A: Japan
Q: Who was Dodge Ram SRT-10 created by? Dodge
Q: Who is the spouse of Joe Biden? A: Jill Biden
Q: Which continent is the country where the director of "My House Husband: Ikaw Na!" was educated located in? A: Asia
Q: What country was the location of the Battle of Pressburg? A: Hungary
Q: Who is the spouse of the US president? A: Jill Biden
Q: Who has ownership of the developer of the Chevrolet Corvette (C4)? A: General Motors
Q: Who is Joe Biden married to? A: Jill Biden
Q: What is the country of citizenship of Charles II of Spain? A: Spain
Q: Who was Chevrolet Biscayne created by? A: Chevrolet
Q: What is the name of the current head of state in United Kingdom? A: Elizabeth II"""

    edit_prompts = {
        "P30": "Q: Which continent is India located in? A: Asia\nQ: Which continent is Canada located in? A: North America\nQ: Which continent is France located in? A: Europe\nQ: Which continent is Sultanate of Egypt located in? A: Africa\nQ: Which continent is Mysore district located in? A: Asia\nQ: Which continent is Germany located in? A: Europe\nQ: Which continent is Renaud Island located in? A: Antarctica\nQ: Which continent is Prince-Bishopric of Warmia located in? A: Europe",
        "P36": "Q: What is the capital of Germany? A: Berlin\nQ: What is the capital of United States of America? A: Washington, D.C.\nQ: What is the capital of Arsanjan County? A: Arsanjan\nQ: What is the capital of Formia? A: Formia\nQ: What is the capital of Grundy County? A: Altamont\nQ: What is the capital of Custer County? A: Arapaho\nQ: What is the capital of France? A: Paris\nQ: What is the capital of India? A: New Delhi",
        "P35": "Q: What is the name of the current head of state in Newfoundland and Labrador? A: Elizabeth II\nQ: What is the name of the current head of state in United States of America? A: Donald Trump\nQ: What is the name of the current head of state in Stoltenberg's Second Cabinet? A: Harald V of Norway\nQ: What is the name of the current head of state in Germany? A: Frank-Walter Steinmeier\nQ: What is the name of the current head of state in India? A: Ram Nath Kovind\nQ: What is the name of the current head of state in Manipur? A: Najma Heptulla\nQ: What is the name of the current head of state in France? A: Emmanuel Macron\nQ: What is the name of the current head of state in Uttarakhand? A: Krishan Kant Paul",
        "P6": "Q: What is the name of the current head of the Winterthur government? A: Michael Künzle\nQ: What is the name of the current head of the Jūrmala government? A: Inese Aizstrauta\nQ: What is the name of the current head of the India government? A: Narendra Modi\nQ: What is the name of the current head of the Saale-Orla-Kreis government? A: Thomas Fügmman\nQ: What is the name of the current head of the Germany government? A: Angela Merkel\nQ: What is the name of the current head of the Bratislava government? A: Matúš Vallo\nQ: What is the name of the current head of the United States of America government? A: Donald Trump\nQ: What is the name of the current head of the France government? A: Édouard Philippe",
        "P20": "Q: Which city did Adolf Hitler die in? A: Führerbunker\nQ: Which city did Ronald Reagan die in? A: Bel Air\nQ: Which city did Abraham Lincoln die in? A: Petersen House\nQ: Which city did William Shakespeare die in? A: Stratford-upon-Avon\nQ: Which city did Eddie Durham die in? A: New York City\nQ: Which city did Anna Howard Shaw die in? A: Nether Providence Township\nQ: Which city did Franz Walter Stahlecker die in? A: Gatchina\nQ: Which city did Antônio Carlos Gomes die in? A: Belém",
        "P26": "Q: Who is Barack Obama married to? A: Michelle Obama\nQ: Who is Elvis Costello married to? A: Diana Krall\nQ: Who is Bill Clinton married to? A: Hillary Clinton\nQ: Who is Mandy Patinkin married to? A: Kathryn Grody\nQ: Who is George W. Bush married to? A: Laura Bush\nQ: Who is John Reed married to? A: Sunday Reed\nQ: Who is Adolf Hitler married to? A: Eva Braun\nQ: Who is John Forrest married to? A: Margaret Forrest",
        "P140": "Q: Which religion is Jaroslav Hašek affiliated with? A: Catholic Church\nQ: Which religion is Catholic Church affiliated with? A: Catholicism\nQ: Which religion is Afghanistan affiliated with? A: Islam\nQ: Which religion is Ottoman Empire affiliated with? A: Islam\nQ: Which religion is Hyder Ali affiliated with? A: Islam\nQ: Which religion is George W. Bush affiliated with? A: United Methodist Church\nQ: Which religion is Mahershala Ali affiliated with? A: Islam\nQ: Which religion is Texas Lutheran University affiliated with? A: American Evangelical Lutheran Church",
        "P1412": "Q: What language does Abraham Lincoln speak? A: English\nQ: What language does Cyril Wecht speak? A: English\nQ: What language does Ronald Reagan speak? A: English\nQ: What language does Augusto Boal speak? A: Portuguese\nQ: What language does William Shakespeare speak? A: English\nQ: What language does Anna Deavere Smith speak? A: English\nQ: What language does George Washington speak? A: English\nQ: What language does Donald Glover speak? A: English",
        "P19": "Q: Which city was Ronald Reagan born in? A: Tampico\nQ: Which city was Adolf Hitler born in? A: Braunau am Inn\nQ: Which city was George W. Bush born in? A: New Haven\nQ: Which city was Akim Tamiroff born in? A: Tbilisi\nQ: Which city was Bill Clinton born in? A: Hope\nQ: Which city was Giovanni Battista Rubini born in? A: Romano di Lombardia\nQ: Which city was Matthew C. Perry born in? A: Newport\nQ: Which city was Matthias Flacius born in? A: Labin",
        "P69": "Q: Which university was Napoleon educated at? A: École Militaire\nQ: Which university was Bob Dylan educated at? A: University of Minnesota\nQ: Which university was Dennis Day educated at? A: Manhattan College\nQ: Which university was Matt Lanter educated at? A: University of Georgia\nQ: Which university was George Villiers, 4th Earl of Clarendon educated at? A: St John's College\nQ: Which university was Anita Rani educated at? A: University of Leeds\nQ: Which university was George Washington educated at? A: Washington and Lee University\nQ: Which university was William Shakespeare educated at? A: King Edward VI School, Stratford-upon-Avon",
        "P40": "Q: Who is Aristotle's child? A: Nicomachus\nQ: Who is Bill Clinton's child? A: Chelsea Clinton\nQ: Who is Elvis Presley's child? A: Lisa Marie Presley\nQ: Who is Grand Duke Michael Alexandrovich of Russia's child? A: George Mikhailovich, Count Brasov\nQ: Who is Crystal Eastman's child? A: Jeffrey Fuller\nQ: Who is Dean Cain's child? A: Christopher Dean Cain\nQ: Who is Hillary Clinton's child? A: Chelsea Clinton\nQ: Who is William Somerset Maugham's child? A: Mary Elizabeth Maugham",
        "P27": "Q: What is the country of citizenship of Bernie Marsden? A: England\nQ: What is the country of citizenship of Bill Clinton? A: United States of America\nQ: What is the country of citizenship of Srinu Vaitla? A: India\nQ: What is the country of citizenship of George W. Bush? A: United States of America\nQ: What is the country of citizenship of José Joaquín Prieto? A: Chile\nQ: What is the country of citizenship of Daniel Razon? A: Philippines\nQ: What is the country of citizenship of Barack Obama? A: United States of America\nQ: What is the country of citizenship of Ronald Reagan? A: United States of America",
        "P175": "Q: Who performed Pete? A: Jim Cummings\nQ: Who performed Cover Version? A: Steven Wilson\nQ: Who performed Spider-man? A: Tom Holland\nQ: Who performed Universal Music Group? A: Helene Fischer\nQ: Who performed Tenth Doctor? A: David Tennant\nQ: Who performed Dear Prudence? A: The Beatles\nQ: Who performed Art Nouveau? A: Giuseppe Amisani\nQ: Who performed Britney Jean? A: Britney Spears",
        "P108": "Q: Who is the employer of Alexander Borodin? A: Saint Petersburg State Medical University\nQ: Who is the employer of Ali Larijani? A: University of Tehran\nQ: Who is the employer of Ronald Reagan? A: Warner Bros.\nQ: Who is the employer of Winston Churchill? A: University of Edinburgh\nQ: Who is the employer of Leon Allen White? A: Impact Wrestling\nQ: Who is the employer of Roger Ebert? A: University of Chicago\nQ: Who is the employer of Madonna? A: Dunkin'\nQ: Who is the employer of Veranke? A: S.H.I.E.L.D.",
        "P112": "Q: Who founded Whitworth Art Gallery? A: Robert Dukinfield Darbishire\nQ: Who founded Chicago? A: Jean Baptiste Point du Sable\nQ: Who founded Israel? A: David Ben-Gurion\nQ: Who founded People's Republic of China? A: Communist Party of China\nQ: Who founded All Ceylon Tamil Congress? A: G. G. Ponnambalam\nQ: Who founded La Boite Theatre Company? A: Jeremiah Joseph Stable\nQ: Who founded Democratic Party? A: Andrew Jackson\nQ: Who founded Hebrew Union College – Jewish Institute of Religion – Cincinnati? A: Isaac Mayer Wise",
        "P50": "Q: Who is the author of Rise and Fall of the City of Mahagonny? A: Bertolt Brecht\nQ: Who is the author of Murder in the Cathedral? A: T. S. Eliot\nQ: Who is the author of Colony in Space? A: Malcolm Hulke\nQ: Who is the author of Dictionary of American Naval Fighting Ships? A: James Mooney\nQ: Who is the author of Manfred? A: Lord Byron\nQ: Who is the author of Hamlet? A: William Shakespeare\nQ: Who is the author of James Bond? A: Ian Fleming\nQ: Who is the author of New Testament? A: various authors",
        "P170": "Q: Who was Twitter created by? A: Jack Dorsey\nQ: Who was Raising the Bar created by? A: Steven Bochco\nQ: Who was Grammy Award created by? A: National Academy of Recording Arts and Sciences\nQ: Who was Facebook created by? A: Mark Zuckerberg\nQ: Who was Remus Lupin created by? A: J. K. Rowling\nQ: Who was Bubbles created by? A: David Simon\nQ: Who was Pilipinas Got Talent created by? A: Simon Cowell\nQ: Who was Finnish created by? A: Mikael Agricola",
        "P407": "Q: Which language was National Register of Historic Places written in? A: English\nQ: Which language was Krazy Kat written in? A: English\nQ: Which language was The Cleveland Show written in? A: English\nQ: Which language was Company written in? A: English\nQ: Which language was AllMusic written in? A: English\nQ: Which language was Billboard Latin Music Awards written in? A: Spanish\nQ: Which language was The New York Times written in? A: English\nQ: Which language was Billboard written in? A: English",
        "P37": "Q: What is the official language of Rabinal? A: Spanish\nQ: What is the official language of Germany? A: German\nQ: What is the official language of France? A: French\nQ: What is the official language of Iran? A: Persian\nQ: What is the official language of Slovak Socialist Republic? A: Slovak\nQ: What is the official language of Akkadian empire? A: Akkadian\nQ: What is the official language of Kingdom of Bavaria? A: German\nQ: What is the official language of United States of America? A: American English",
        "P740": "Q: Where was Israel founded? A: Independence Hall\nQ: Where was The Guardian founded? A: Manchester\nQ: Where was The Bevis Frond founded? A: London\nQ: Where was People's Republic of China founded? A: Tiananmen\nQ: Where was Thomson Reuters founded? A: Toronto\nQ: Where was Burzum founded? A: Norway\nQ: Where was Florentine Opera founded? A: Wisconsin\nQ: Where was NBC founded? A: New York City",
        "P495": "Q: Which country was Elfquest created in? A: United States of America\nQ: Which country was The New York Times created in? A: United States of America\nQ: Which country was Drum Beat created in? A: United States of America\nQ: Which country was association football created in? A: England\nQ: Which country was basketball created in? A: United States of America\nQ: Which country was cricket created in? A: England\nQ: Which country was Jake and the Never Land Pirates created in? A: United States of America\nQ: Which country was PC Pro created in? A: United Kingdom",
        "P106": "Q: What kind of work does Diori Hamani do? A: politician\nQ: What kind of work does William Shakespeare do? A: playwright\nQ: What kind of work does Keith Law do? A: editor\nQ: What kind of work does Elizabeth II do? A: monarch\nQ: What kind of work does Fred Davis do? A: snooker player\nQ: What kind of work does Adolf Hitler do? A: statesperson\nQ: What kind of work does Tom Richardson do? A: cricketer\nQ: What kind of work does Stephan von Breuning do? A: entomologist",
        "P136": "Q: What type of music does Bible play? A: religious text\nQ: What type of music does Radney Foster play? A: country music\nQ: What type of music does Columbia Records play? A: heavy metal\nQ: What type of music does Pietro Mascagni play? A: opera\nQ: What type of music does De revolutionibus orbium coelestium play? A: treatise\nQ: What type of music does Stanley Myers play? A: film score\nQ: What type of music does IOWA play? A: independent film\nQ: What type of music does William Shakespeare play? A: English Renaissance theatre",
        "P364": "Q: What is the original language of IOWA? A: English\nQ: What is the original language of Talent Scout? A: English\nQ: What is the original language of Doctor Who? A: English\nQ: What is the original language of Saturday Night Live? A: English\nQ: What is the original language of Trials and Tribble-ations? A: English\nQ: What is the original language of Wolf Creek? A: English\nQ: What is the original language of Reservoir Dogs? A: English\nQ: What is the original language of The Simpsons? A: English",
        "P937": "Q: Which city did Franklin Delano Roosevelt work in? A: Washington, D.C.\nQ: Which city did Jesus Christ work in? A: Galilee\nQ: Which city did Stanley Bruce work in? A: Canberra\nQ: Which city did Luis Gutiérrez work in? A: Washington, D.C.\nQ: Which city did Bill Clinton work in? A: Washington, D.C.\nQ: Which city did John Spencer, 3rd Earl Spencer work in? A: London\nQ: Which city did John F. Kennedy work in? A: Washington, D.C.\nQ: Which city did Henry Scott, 3rd Duke of Buccleuch work in? A: London",
        "P800": "Q: What is John Lennon famous for? A: Imagine\nQ: What is Muhammad famous for? A: Quran\nQ: What is Brian Ferneyhough famous for? A: String Quartet No. 2\nQ: What is Elton John famous for? A: Billy Elliot the Musical\nQ: What is John Webb famous for? A: Wilton House\nQ: What is Marina Abramović famous for? A: Seven Easy Pieces\nQ: What is Edward Gibbon famous for? A: The History of the Decline and Fall of the Roman Empire\nQ: What is Adolf Hitler famous for? A: Mein Kampf",
        "P641": "Q: Which sport is handball at the 1972 Summer Olympics associated with? A: handball\nQ: Which sport is Major League Baseball associated with? A: baseball\nQ: Which sport is Scottish Premiership associated with? A: rugby union\nQ: Which sport is first-class cricket associated with? A: cricket\nQ: Which sport is Willenhall Town F.C. associated with? A: association football\nQ: Which sport is National Football League associated with? A: American football\nQ: Which sport is college football associated with? A: American football\nQ: Which sport is Ed Delahanty associated with? A: baseball",
        "P413": "Q: What position does Steven Naismith play? A: forward\nQ: What position does Hugo Chávez play? A: pitcher\nQ: What position does Juan Cuadrado play? A: fullback\nQ: What position does Grant Fuhr play? A: goaltender\nQ: What position does Babe Ruth play? A: left fielder\nQ: What position does R. Kelly play? A: shooting guard\nQ: What position does Gerald Ford play? A: center\nQ: What position does Yu Darvish play? A: starting pitcher",
        "P286": "Q: Who is the head coach of Manchester United F.C.? A: Ole Gunnar Solskjær\nQ: Who is the head coach of Hideo Itami? A: Kenta Kobashi\nQ: Who is the head coach of England national football team? A: Gareth Southgate\nQ: Who is the head coach of Tony Nese? A: Mikey Whipwreck\nQ: Who is the head coach of Spain national football team? A: Robert Moreno\nQ: Who is the head coach of Liverpool F.C.? A: Dario Bonetti\nQ: Who is the head coach of New York Yankees? A: Aaron Boone\nQ: Who is the head coach of US Créteil-Lusitanos? A: Carlos Secretário",
        "P159": "Q: Which city is the headquarter of Sestao River Club located in? A: Sestao\nQ: Which city is the headquarter of National Register of Historic Places located in? A: Washington, D.C.\nQ: Which city is the headquarter of Centre for Humanitarian Dialogue located in? A: Geneva\nQ: Which city is the headquarter of Douglas located in? A: Kingswood\nQ: Which city is the headquarter of Republican Party located in? A: Washington, D.C.\nQ: Which city is the headquarter of Travel Air located in? A: Wichita\nQ: Which city is the headquarter of Democratic Party located in? A: Washington, D.C.\nQ: Which city is the headquarter of United States Census Bureau located in? A: Suitland",
        "P178": "Q: Who is the developer of Telegram? A: Telegram FZ-LLC\nQ: Who is the developer of Microsoft Windows? A: Microsoft\nQ: Who is the developer of PlayStation 2? A: Sony Interactive Entertainment\nQ: Who is the developer of iTunes? A: Apple Inc.\nQ: Who is the developer of SR-71 Blackbird? A: Kelly Johnson\nQ: Who is the developer of Moblin? A: Linux Foundation\nQ: Who is the developer of Xbox 360? A: Microsoft\nQ: Who is the developer of Kinsey scale? A: Alfred Kinsey",
        "P488": "Q: Who is the chairperson of Borussia Mönchengladbach? A: Rolf Königs\nQ: Who is the chairperson of Senate? A: Gérard Larcher\nQ: Who is the chairperson of Harvard University? A: Lawrence S. Bacow\nQ: Who is the chairperson of Republican Party? A: Ronna Romney McDaniel\nQ: Who is the chairperson of Chinese People's Political Consultative Conference? A: Yu Zhengsheng\nQ: Who is the chairperson of Bangladesh Bank? A: Fazle Kabir\nQ: Who is the chairperson of Democratic Party? A: Thomas Perez\nQ: Who is the chairperson of United States House of Representatives? A: Nancy Pelosi",
        "P169": "Q: Who is the chief executive officer of Poste italiane? A: Luisa Todini\nQ: Who is the chief executive officer of HBO? A: Richard Plepler\nQ: Who is the chief executive officer of Boston Market? A: George Michel\nQ: Who is the chief executive officer of Cendant? A: Henry Silverman\nQ: Who is the chief executive officer of Twitter? A: Jack Dorsey\nQ: Who is the chief executive officer of YouTube? A: Susan Wojcicki\nQ: Who is the chief executive officer of The Royal Bank of Scotland? A: Ross McEwan\nQ: Who is the chief executive officer of Microsoft? A: Satya Nadella",
        "P449": "Q: Who is the original broadcaster of American Idol? A: Fox Broadcasting Company\nQ: Who is the original broadcaster of The Cisco Kid? A: broadcast syndication\nQ: Who is the original broadcaster of USA Today? A: broadcast syndication\nQ: Who is the original broadcaster of Chase? A: NBC\nQ: Who is the original broadcaster of Saturday Night Live? A: NBC\nQ: Who is the original broadcaster of Fairly Legal? A: USA Network\nQ: Who is the original broadcaster of The Simpsons? A: Fox Broadcasting Company\nQ: Who is the original broadcaster of Republic of Doyle? A: CBC Television",
        "P176": "Q: Which company is PlayStation 2 produced by? A: Sony Interactive Entertainment\nQ: Which company is Mercury-Redstone 3 produced by? A: McDonnell Aircraft\nQ: Which company is Altair 8800 produced by? A: Micro Instrumentation and Telemetry Systems\nQ: Which company is PlayStation 3 produced by? A: Sony Interactive Entertainment\nQ: Which company is Alcantara produced by? A: Alcantara\nQ: Which company is Dodge Dakota produced by? A: Dodge\nQ: Which company is carbon dioxide produced by? A: humanity\nQ: Which company is United States dollar produced by? A: Bureau of Engraving and Printing",
        "P1037": "Q: Who is the director of Athens Conservatoire? A: George Nazos\nQ: Who is the director of World Economic Forum? A: Klaus Schwab\nQ: Who is the director of National Hockey League? A: Gary Bettman\nQ: Who is the director of Opéra-Comique? A: Olivier Mantei\nQ: Who is the director of Fraunhofer Society? A: Reimund Neugebauer\nQ: Who is the director of American Broadcasting Company? A: Bob Iger\nQ: Who is the director of National Aeronautics and Space Administration? A: Jim Bridenstine\nQ: Who is the director of British Broadcasting Corporation? A: Tony Hall, Baron Hall of Birkenhead",
        "P1308": "Q: Who is the President of the United States? A: Donald Trump\nQ: Who is the monarch of Italy? A: Odoacer\nQ: Who is the President pro tempore of the United States Senate? A: Orrin Hatch\nQ: Who is the Prime Minister of the United Kingdom? A: Boris Johnson\nQ: Who is the Governor of Tennessee? A: Bill Haslam\nQ: Who is the Vice President of the United States? A: Mike Pence\nQ: Who is the Premier of North Korea? A: Kim Jae-ryong\nQ: Who is the pope? A: Francis"
    }

    main(
        model_name="EleutherAI/gpt-j-6B",
        ds_name="mquake",
        dataset_size_limit=3000,
        generation_test_interval=1,
        dir_name="your_results_dir",
        multi_hop_prompt=multi_hop_prompt,
        edit_prompts=edit_prompts
    )
