import typing
import nltk
import numpy as np
import scipy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity
import re

def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


import torch
import typing
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import shutil
import os

def clear_torch_cache():
    cache_dir = os.path.expanduser('~/.cache/torch/kernels')
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Failed to remove {item_path}. Reason: {e}")
        
def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    snips: typing.Optional[AttributeSnippets] = None,
    vec: typing.Optional[TfidfVectorizer] = None
) -> typing.Dict:
    """
    Evaluates the rewritten model on a MQuAKE dataset record for multiple metrics including
    edit-wise success rate, instance-wise accuracy, and multi-hop accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :param snips: Optional, attribute snippets for reference texts.
    :param vec: Optional, a TF-IDF vectorizer.
    :return: A dictionary with evaluation metrics.
    """
    multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers = calculate_metrics(
        model, tokenizer, record
    )

    # 打印各个指标
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

def calculate_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict
):
    """
    Calculate multi-hop accuracy, edit-wise success rate, and instance-wise accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :return: Multi-hop accuracy, edit-wise success rate, instance-wise accuracy, and generated answers.
    """
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

    correct_responses = 0
    success_count = 0
    all_facts_recalled = True
    generated_answers = []

    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    extended_answers = record.get('answer_extended', [])
    requested_rewrite = record['requested_rewrite']

    for question in questions:
        # 使用 generate_fast 函数生成答案
        full_prompt = multi_hop_prompt + "\n" + question
        generated_text = generate_fast(model, tokenizer, [full_prompt], n_gen_per_prompt=1, max_out_len=100)[0]
        generated_answer = generated_text
        
        generated_answers.append(generated_answer)

        # Debugging information
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")

        if (correct_answer.lower() in generated_answer.lower() or 
            any(alias.lower() in generated_answer.lower() for alias in answer_aliases) or 
            any(answer.lower() in generated_answer.lower() for answer in extended_answers)):
            correct_responses += 1

    for rewrite in requested_rewrite:
        question = rewrite['prompt'].format(rewrite['subject'])
        prompt_key = rewrite['prompt_key']
        full_prompt = edit_prompts[prompt_key] + "\n" + question
        generated_text = generate_fast(model, tokenizer, [full_prompt], n_gen_per_prompt=1, max_out_len=100)[0]
        generated_answer = generated_text
        
        generated_answers.append(generated_answer)

        print(f"Rewrite Question: {question}")
        print(f"Generated Answer: {generated_answer}")

        target_new = rewrite['target_new']['str']

        if target_new.lower() in generated_answer.lower():
            success_count += 1

    # Check if all facts are recalled
    all_facts_recalled = (success_count == len(requested_rewrite))

    multi_hop_accuracy = correct_responses / len(questions)
    edit_success_rate = success_count / len(requested_rewrite)
    instance_accuracy = 1 if all_facts_recalled else 0

    return multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers

'''import unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def calculate_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict
):
    """
    Calculate multi-hop accuracy, edit-wise success rate, and instance-wise accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :return: Multi-hop accuracy, edit-wise success rate, instance-wise accuracy, and generated answers.
    """
    correct_responses = 0
    success_count = 0
    all_facts_recalled = True
    generated_answers = []

    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    requested_rewrite = record['requested_rewrite']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取模型所在的设备
    model.to(device)  # 确保模型在 GPU 上
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 设置 pad_token_id

    for question in questions + [rw['prompt'].format(rw['subject']) for rw in requested_rewrite]:
        # 使用 model.generate 函数生成答案
        inputs = tokenizer([question], padding=True, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 生成时调试信息
        print(f"input_ids device: {input_ids.device}, attention_mask device: {attention_mask.device}")
        clear_torch_cache
        # Generate outputs using model.generate
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            do_sample=True,
            top_k=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode the generated outputs
        generated_texts = [
            tokenizer.decode(output.cpu(), skip_special_tokens=True).split(tokenizer.eos_token)[0].strip()
            for output in generated_outputs
        ]

        # Normalize the text
        generated_texts = [
            unicodedata.normalize("NFKD", text).replace("\n\n", " ").replace("\n", " ").replace("", "")
            for text in generated_texts
        ]

        generated_answer = generated_texts[0]

        # 获取生成文本的回答部分，针对 multi-hop accuracy
        if question in questions:
            generated_answers.append(generated_answer)

            # Debugging information
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_answer}")

            if correct_answer.lower() in generated_answer.lower() or any(alias.lower() in generated_answer.lower() for alias in answer_aliases):
                correct_responses += 1

        # 针对 edit-wise success rate 和 instance-wise accuracy
        if question not in questions:
            matching_rewrites = [rw for rw in requested_rewrite if rw['prompt'].format(rw['subject']) == question]
            
            if not matching_rewrites:
                print(f"No matching rewrite found for question: {question}")
                continue
            
            target_new = matching_rewrites[0]['target_new']['str']

            if target_new.lower() in generated_answer.lower():
                success_count += 1

    # Check if all facts are recalled
    all_facts_recalled = (success_count == len(requested_rewrite))

    multi_hop_accuracy = correct_responses / len(questions)
    edit_success_rate = success_count / len(requested_rewrite)
    instance_accuracy = 1 if all_facts_recalled else 0

    return multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers'''














def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()

# 在这里添加你的评估代码







