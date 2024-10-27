import pandas as pd
import nltk
import numpy as np
from datasets import load_dataset, load_from_disk
import os
from utils import *
import jsonlines
import json
from torch.utils.data import Dataset

class SCOTUS(Dataset):
    """
    Dataset class for handling SCOTUS data.
    
    Args:
        split (str): 'train' or 'edit' (test).
    """
    def __init__(self, split):
        try:
            if split == "train":
                data = load_dataset("tomh/grace-scotus", split="train")
            elif split == "edit":
                data = load_dataset("tomh/grace-scotus", split="test")
            text = data['text']
            labels = data['label']
            self.data = [{"text": x, "labels": y} for x, y in zip(text, labels)]
        except Exception as e:
            print(f"Error loading SCOTUS dataset: {str(e)}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NQ(Dataset):
    """
    Dataset class for Natural Questions (NQ).
    
    Args:
        path (str): Path to the NQ data file.
    """
    def __init__(self, path="./grace/data/nq_train.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                NQ = json.load(f)
            questions, answers = NQ["questions"], NQ["answers"]
            self.data = [{"text": x, "labels": y} for x, y in zip(questions[:1000], answers[:1000])]
        except Exception as e:
            print(f"Error loading NQ dataset: {str(e)}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class zsRE(Dataset):
    """
    Dataset class for Zero-shot RE (zsRE) data.
    
    Args:
        path (str): Path to the zsRE data file in JSON lines format.
        split (str): The data split to load ('edit' or 'holdout').
    """
    def __init__(self, path="./grace/data/structured_zeroshot-dev-new_annotated_final.jsonl", split="edit"):
        try:
            questions, answers = self.load_zsre(path)
            edits = [{"text": x, "labels": y} for x, y in zip(questions, answers)]
            n_edits = min(10000, len(questions))
            np.random.seed(42)
            shuffle_ix = np.random.choice(n_edits, n_edits, replace=False)
            shuffle_edit, shuffle_holdout = shuffle_ix[:(n_edits // 2)], shuffle_ix[(n_edits // 2):]
            edit_batches = [edits[i] for i in shuffle_edit]
            edit_batches_holdout = [edits[i] for i in shuffle_holdout]

            print(f"Loaded {len(edit_batches)} possible edits and {len(edit_batches_holdout)} holdouts.")

            if split == "edit":
                self.data = edit_batches
            elif split == "holdout":
                self.data = edit_batches_holdout
            else:
                print(f"split '{split}' undefined")
                self.data = []
        except Exception as e:
            print(f"Error loading zsRE dataset: {str(e)}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_zsre(self, data_path):
        questions, answers = [], []
        try:
            with jsonlines.open(data_path, encoding="utf-8") as f:
                for d in f:
                    ex = {k: d[k] for k in ["input", "prediction", "alternatives", "filtered_rephrases", "output"]}
                    questions.append(ex["input"])
                    answers.append(ex["output"][0]["answer"])
                    if len(ex["filtered_rephrases"]) >= 10:
                        for rephrase in ex["filtered_rephrases"][:10]:
                            questions.append(rephrase)
                            answers.append(ex["output"][0]["answer"])
        except Exception as e:
            print(f"Error reading zsRE data: {e}")
        return questions, answers


class WebText10k(Dataset):
    """
    Dataset class for OpenWebText-10k data.
    """
    def __init__(self):
        try:
            data = load_dataset('stas/openwebtext-10k')['train']
            upstream = data["text"][:1000]
            self.text = [{"text": s, "labels": [], "concept": []} for s in upstream]
        except Exception as e:
            print(f"Error loading WebText10k dataset: {str(e)}")
            self.text = []

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]


class Hallucination(Dataset):
    """
    Dataset class for handling hallucination data in Wikipedia passages.
    
    Args:
        split (str): The data split to load ('edit', 'accurate', 'original', or 'pretrain').
        concept_path (str): Path to the concept file (optional).
    """
    def __init__(self, split, concept_path=None):
        self.concept_path = concept_path if concept_path else './grace/wiki_bio_concepts.txt'
        try:
            self.data = pd.DataFrame(load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"])
            concepts = self.load_concepts(self.concept_path)
            self.concepts = [s.strip() for s in concepts]
            edit_batches, accurates, originals = self.get_edits(self.data, self.concepts)
        except Exception as e:
            print(f"Error initializing Hallucination dataset: {str(e)}")
            self.text = []
            return

        if split == "edit":
            self.text = edit_batches
            print(f"Loaded {len(self.text)} edits")
        elif split == "accurate":
            self.text = accurates
            print(f"Loaded {len(self.text)} accurates")
        elif split == "original":
            self.text = originals
            print(f"Loaded {len(self.text)} originals")
        elif split == "pretrain":
            upstream = WebText10k()
            self.text = accurates + originals + upstream.text[:200]
            self.text = [{"text": x["text"], "labels": [], "concept": []} for x in self.text]
            print(f"Loaded {len(self.text)} pretraining instances")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def load_concepts(self, PATH):
        try:
            if not os.path.exists(PATH):
                concepts = self.generate_concepts()
            else:
                with open(PATH, 'r', encoding="utf-8") as f:  # UTF-8 encoding specified here
                    concepts = f.readlines()

            # Regenerate if existing concepts are different in shape (this dataset keeps getting updated)
            if len(concepts) != len(self.data):
                concepts = self.generate_concepts()
            return concepts
        except Exception as e:
            print(f"Error loading concepts: {str(e)}")
            return []


    def generate_concepts(self):
        try:
            wikibio = load_dataset("wiki_bio")
            bio_idx = self.data["wiki_bio_test_idx"]
            concepts = [wikibio["test"]["input_text"][i]["context"].strip().replace("-lrb- ", "").replace(" -rrb-", "")
                        for i in bio_idx]
            with open(self.concept_path, 'w', encoding="utf-8") as f:
                f.write('\n'.join(concepts))
            return concepts
        except Exception as e:
            print(f"Error generating concepts: {str(e)}")
            return []

    def get_edits(self, data, concepts):
        edits, originals, accurates = [], [], []
        try:
            for i in range(len(self.data)):
                header = f"This is a Wikipedia passage about {concepts[i]}."
                annotations = self.data["annotation"][i]
                correct_sentences = nltk.sent_tokenize(self.data["wiki_bio_text"][i])[:len(annotations)]
                for j, annotation in enumerate(annotations):
                    prompt = " ".join(self.data["gpt3_sentences"][i][:j])
                    if "inaccurate" in annotation:
                        edits.append({
                            "text": f"{header} {prompt}",
                            "labels": correct_sentences[min(j, len(correct_sentences) - 1)],
                            "concept": concepts[i],
                        })
                        originals.append({
                            "text": f"{header} {prompt}",
                            "labels": self.data["gpt3_sentences"][i][j],
                            "concept": concepts[i],
                        })
                    else:
                        accurates.append({
                            "text": f"{header} {prompt}",
                            "labels": self.data["gpt3_sentences"][i][j],
                            "concept": concepts[i],
                        })
        except Exception as e:
            print(f"Error generating edits: {str(e)}")
        return edits, accurates, originals
