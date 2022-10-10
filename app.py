from crypt import methods
from distutils.log import error
from utils import spell_checker
from flask import Flask, render_template, request, jsonify
from corrector import ScRNNChecker

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index(diffs=None):
    language = request.args.get("language")
    task_name = request.args.get("error-channel")
    sentence_with_errors = request.args.get("sentence")
    sentence_without_errors = ""

    if sentence_with_errors:
        sentence_without_errors = correct_sentence(language, task_name, sentence_with_errors)
        diffs = spell_checker(sentence_with_errors, sentence_without_errors)

    error_channels = get_error_channels_by_language(language) if language else []
    data = {"language": language, "error_channel": task_name, "error_channels": error_channels,
            "sentence": sentence_without_errors, "diffs": diffs}

    return render_template("index.html", data=data)


def correct_sentence(language, task_name, sentence_with_errors):
    if task_name == "ensamble":
        print("ensamble")
        prediction = ensamble(language, sentence_with_errors)
    else:
        checker = ScRNNChecker(language=language, task_name=task_name)
        prediction = checker.correct_string(sentence_with_errors)

    return prediction


def ensamble(language, sentence_with_errors):
    predictions = []
    checker = ScRNNChecker(language=language, task_name="unnorm_teacher_general")
    prediction = checker.correct_string(sentence_with_errors)
    predictions.append(prediction)

    if language != "yi":
        checker = ScRNNChecker(language=language, task_name="phongrafamb_teacher_general")
        prediction = checker.correct_string(sentence_with_errors)
        predictions.append(prediction)

    checker = ScRNNChecker(language=language, task_name="keyprox_teacher_general")
    prediction = checker.correct_string(sentence_with_errors)
    predictions.append(prediction)

    checker = ScRNNChecker(language=language, task_name="random_teacher_general")
    prediction = checker.correct_string(sentence_with_errors)
    predictions.append(prediction)

    checker = ScRNNChecker(language=language, task_name="sylsim_teacher_general")
    prediction = checker.correct_string(sentence_with_errors)
    predictions.append(prediction)

    word_length = len(sentence_with_errors.split(" "))
    words_prediction_voting = []
    for idx in range(word_length):
        voting_dict = {}
        for sentence_prediction in predictions:
            word = sentence_prediction.split(" ")[idx]
            if word not in voting_dict:
                voting_dict[word] = 1
            else:
                voting_dict[word] += 1

        sorted_voting_dict = sorted(voting_dict, key=voting_dict.get, reverse=True)
        words_prediction_voting.append(list(sorted_voting_dict)[0])

    sentence_predicted_voting = " ".join(words_prediction_voting)

    return sentence_predicted_voting


@app.route("/get_error_channels", methods=["POST", "GET"])
def get_error_channels():
    language = request.args.get("language")
    error_channels = get_error_channels_by_language(language)

    return jsonify(error_channels)


def get_error_channels_by_language(language):
    if language == "shi" or language == "ash" or language == "ya":
        error_channels = [
            {"value": "random_teacher_general", "text": "Error Aleatorio"},
            {"value": "keyprox_teacher_general", "text": "Proximidad de Teclado"},
            {"value": "phongrafamb_teacher_general", "text": "Ambigüedad Fonema Grafema"},
            {"value": "sylsim_teacher_general", "text": "Similitud de Sílaba"},
            {"value": "unnorm_teacher_general", "text": "Desnormalización"},
            {"value": "phongrafamb_unnorm_teacher_general", "text": "Ambigüedad Fonema Grafema + Desnormalización"},
            {"value": "all_teacher_general", "text": "Todos"},
            {"value": "ensamble", "text": "Ensamble"}
        ]
    else:
        error_channels = [
            {"value": "random_teacher_general", "text": "Error Aleatorio"},
            {"value": "keyprox_teacher_general", "text": "Proximidad de Teclado"},
            {"value": "sylsim_teacher_general", "text": "Similitud de Sílaba"},
            {"value": "unnorm_teacher_general", "text": "Desnormalización"},
            {"value": "all_teacher_general", "text": "Todos"},
            {"value": "ensamble", "text": "Ensamble"}
        ]

    return error_channels


@app.route("/api/corregir", methods=["GET"])
def corregir():
    language = request.args.get("language")
    task_name = request.args.get("model")
    sentence_with_errors = request.args.get("sentence")

    sentence_without_errors = correct_sentence(language, task_name, sentence_with_errors)
    diffs = spell_checker(sentence_with_errors, sentence_without_errors)

    data = {"corrected_sentence": sentence_without_errors, "diffs": diffs}
    return jsonify({"data": data})


if __name__ == "__main__":
    app.run(debug=True)
