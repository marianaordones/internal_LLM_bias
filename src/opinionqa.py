"""OpinionQA loading and prompt construction."""

import json
import string


def load_questions(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)

    questions = []
    for qkey, record in data.items():
        options = list(record["options"])
        # Refusal is metadata in this dataset, not a response option for the model.
        if options and options[-1] == "Refused":
            options = options[:-1]
        if len(options) >= 2:
            questions.append({
                "qkey": qkey,
                "question": record["refined_qbody"],
                "options": options,
            })
    return questions


def build_user_message(question, options, declaration=None):
    letters = string.ascii_uppercase[:len(options)]
    option_text = "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, options)
    )
    prefix = f"{declaration}\n\n" if declaration else ""
    return (
        f"{prefix}{question.strip()}\n\n{option_text}\n\n"
        "Answer with the single letter of the option that best matches your view.\n\n"
        "Answer:"
    ), letters


def build_named_user_message(name, question, options):
    name_sentence = f"Hi! My name is {name}."
    # Reuse the declaration slot so prompt spacing matches the other experiments.
    message, letters = build_user_message(question, options, name_sentence)
    return message, name_sentence, letters
