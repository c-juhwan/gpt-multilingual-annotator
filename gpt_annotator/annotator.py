import os
import json
import random
import logging
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import openai
from tqdm.auto import tqdm
openai.api_key = os.environ["OPENAI_API_KEY"]
tqdm_bar = tqdm(total=100, desc="Annotation Progress", position=0)

def main(args: argparse.Namespace) -> None:
    # Load input json data
    with open(args.input, "r") as f:
        file = json.load(f)
    config = file["config"]
    data = file["data"]
    # Update tqdm bar
    global tqdm_bar
    tqdm_bar.reset(total=len(data))

    # Divide data into num_processes chunks
    data_chunks = []
    for i in range(args.num_processes):
        data_chunks.append(data[i::args.num_processes])

    # Call multiprocessing
    starmap_items = [
        (args, config, data_chunks[i]) for i in range(args.num_processes)
    ]

    tqdm.write(f"Start multiprocessing with {args.num_processes} processes")
    tqdm.write(f"Total number of data: {len(data)}")

    with Pool(args.num_processes) as p:
        results = p.starmap(try_call_gpt, starmap_items)

    tqdm.write("Multiprocessing finished")
    tqdm_bar.close()

    # Merge results
    merged_results = file
    merged_results["data"] = [] # clear data
    for each_result in results:
        print(len(each_result))
        merged_results["data"].extend(each_result)
    if config['task'] == 'image_captioning':
        assert len(merged_results["data"]) == len(data)
    elif config['task'] == 'text_style_transfer':
        assert len(merged_results["data"]) == len(data) * 2

    # Sort data by idx
    merged_results["data"].sort(key=lambda x: x["idx"])

    # Save results
    with open(args.output, "w") as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)
    tqdm.write("Saved results")

def try_call_gpt(args: argparse.Namespace, config: dict, data: list) -> None:
    try:
        if config['task'] == "image_captioning":
            return call_gpt_captioning(args, config, data)
        elif config['task'] == 'text_style_transfer':
            return call_gpt_text_style_transfer(args, config, data)
    except KeyboardInterrupt as k:
        raise k
    except Exception as e:
        logging.exception(f"Error in try_call_gpt: {e}")

def call_gpt_captioning(args: argparse.Namespace, config: dict, data: list) -> list:
    return_list = []

    for idx in range(len(data)):
        return_list.append(data[idx])
        # Get source caption to be annotated
        if args.random_selection != 0:
            # Randomly select source caption
            src_caption = random.choice(data[idx]["source_captions"])
        else:
            # Select the first source caption
            src_caption = data[idx]["source_captions"][0]

        # Remove last user input from prompt and add new user input
        config['prompt'].pop()
        config['prompt'].append({"role": "user", "content": f"Input: {src_caption}"})

        error_counter = 0
        while True:
            try:
                # Get gpt paraphrases
                gpt_response = openai.ChatCompletion.create(
                    model=config['gpt_model_version'],
                    messages=config['prompt'],
                )

                # Break down the response into sentences
                gpt_sentences = gpt_response['choices'][0]['message']['content'].split("\n")
                # Remove the ~: part
                for i in range(len(gpt_sentences)):
                    # Remove multiple spaces
                    gpt_sentences[i] = " ".join(gpt_sentences[i].split())
                    # Remove the ~: part
                    gpt_sentences[i] = gpt_sentences[i][gpt_sentences[i].find(":") + 2:]
                # Remove empty strings
                gpt_sentences = list(filter(None, gpt_sentences))

                result_sentences = []
                result_sentences.append(gpt_sentences[0])
                for i in range(1, len(gpt_sentences)):
                    result_sentences.append(gpt_sentences[i].split(" / ")[1])
            except Exception as e:
                tqdm.write(str(e))
                error_counter += 1
                if error_counter > args.error_patience:
                    tqdm.write(f"Error: Too many errors. Skip this image.")
                    break
                continue

            if len(result_sentences) == 5:
                # if gpt_response is correctly generated and has 5 sentences
                break # break the while loop
            else:
                # if gpt_response is not correctly generated, print error message and try again
                error_counter += 1
                if error_counter >= args.error_patience:
                    tqdm.write(f"Error: Too many errors. Skip this image.")
                    break
                continue
        if error_counter >= args.error_patience:
            continue # skip this image

        # Add gpt paraphrases to return_list
        return_list[idx]["target_silver_captions"] = result_sentences

        if tqdm_bar.n + args.num_processes <= tqdm_bar.total:
            tqdm_bar.update(args.num_processes)

    return return_list

def call_gpt_text_style_transfer(args: argparse.Namespace, config: dict, data: list) -> list:
    return_list = []

    for idx in range(0, len(data)*2, 2): # double the idx -> we will generate two paraphrases for each data
        return_list.append(data[idx//2].copy())
        return_list.append(data[idx//2].copy())

        return_list[idx]['idx'] = return_list[idx]['idx'] * 2
        return_list[idx+1]['idx'] = return_list[idx+1]['idx'] * 2 + 1

        src_informal = data[idx//2]["source_informal"]
        src_formal = data[idx//2]["source_formal"]

        # Remove last user input from prompt and add new user input
        config['prompt'].pop()
        config['prompt'].append({"role": "user", "content": f"[Input Sentence]\n\
Formal 1: {src_formal}\n\
Informal 1: {src_informal}"})

        error_counter = 0
        while True:
            try:
                # Get gpt paraphrases
                gpt_response = openai.ChatCompletion.create(
                    model=config['gpt_model_version'],
                    messages=config['prompt'],
                )

                gpt_sentences = gpt_response['choices'][0]['message']['content']

                # Break down response into two part
                paraphrases = gpt_sentences.split('\n\n')[0]
                translations = gpt_sentences.split('\n\n')[1]

                # We only use translations
                translations = translations.split('\n')
                if len(translations) != 5:
                    # if translations is not correctly generated, print error message and try again
                    raise ValueError("Error: translation is not correctly generated")

                formal_1 = translations[1].split('Formal 1: ')[1]
                informal_1 = translations[2].split('Informal 1: ')[1]
                formal_2 = translations[3].split('Formal 2: ')[1]
                informal_2 = translations[4].split('Informal 2: ')[1]

                break # if gpt_response is correctly generated, break the while loop
            except Exception as e:
                tqdm.write(str(e))
                error_counter += 1
                if error_counter > args.error_patience:
                    tqdm.write(f"Error: Too many errors. Skip this data.")
                    break
                continue

        print(formal_1, informal_1, formal_2, informal_2)

        return_list[idx]["target_formal"] = formal_1
        return_list[idx]["target_informal"] = informal_1
        return_list[idx+1]["target_formal"] = formal_2
        return_list[idx+1]["target_informal"] = informal_2

        if tqdm_bar.n + args.num_processes <= tqdm_bar.total:
            tqdm_bar.update(args.num_processes)

    return return_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--random_selection", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--error_patience", type=int, default=5)

    args = parser.parse_args()

    if args.random_selection != 0:
        random.seed(args.random_selection)

    main(args)
