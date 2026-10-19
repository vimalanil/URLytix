
from flask import Flask, render_template, request
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
import joblib
import re

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "urlytix_linear_svm.pkl"

# Load the trained pipeline once when the application starts.
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]


def extract_url_features(url):
    """Must match the extractor used to train the saved model."""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    url_length = len(url)
    domain_length = len(domain)

    is_domain_ip = int(
        re.fullmatch(
            r"\d{1,3}(?:\.\d{1,3}){3}", domain
        ) is not None
    )

    domain_parts = domain.split(".")
    tld = domain_parts[-1] if len(domain_parts) >= 2 else ""
    tld_length = len(tld)
    subdomains = max(len(domain_parts) - 2, 0)

    encoded_chars = re.findall(r"%[0-9a-fA-F]{2}", url)
    obfuscated_count = len(encoded_chars)
    has_obfuscation = int(obfuscated_count > 0)

    obfuscation_ratio = (
        obfuscated_count / url_length if url_length else 0
    )

    letters = sum(c.isalpha() for c in url)
    letter_ratio = letters / url_length if url_length else 0

    digits = sum(c.isdigit() for c in url)
    digit_ratio = digits / url_length if url_length else 0

    equals_count = url.count("=")
    question_count = url.count("?")
    ampersand_count = url.count("&")

    special_count = sum(not c.isalnum() for c in url)
    special_ratio = special_count / url_length if url_length else 0

    is_https = int(parsed.scheme.lower() == "https")

    return [
        url_length,
        domain_length,
        is_domain_ip,
        tld_length,
        subdomains,
        has_obfuscation,
        obfuscated_count,
        obfuscation_ratio,
        letters,
        letter_ratio,
        digits,
        digit_ratio,
        equals_count,
        question_count,
        ampersand_count,
        special_count,
        special_ratio,
        is_https
    ]


def validate_url(raw_url):
    """Basic input validation; this does not verify URL safety."""

    raw_url = raw_url.strip()

    if not raw_url:
        return None, "Please enter a URL."

    if any(char.isspace() for char in raw_url):
        return None, "The URL must not contain spaces."

    if len(raw_url) > 2048:
        return None, "The URL is too long."

    candidate = raw_url
    if not candidate.startswith(("http://", "https://")):
        candidate = "http://" + candidate

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname

        # Accessing port can raise ValueError for invalid ports.
        _ = parsed.port

        if parsed.scheme not in ("http", "https") or not hostname:
            return None, "Enter a valid HTTP or HTTPS URL."

    except ValueError:
        return None, "Enter a valid URL."

    return candidate, None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_url = ""
    decision_score = None

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()
        normalized_url, error = validate_url(submitted_url)

        if not error:
            try:
                values = extract_url_features(normalized_url)

                features_df = pd.DataFrame(
                    [values],
                    columns=FEATURE_NAMES
                )

                prediction = int(model.predict(features_df)[0])
                decision_score = float(
                    model.decision_function(features_df)[0]
                )

                if prediction == 0:
                    result = "PHISHING"
                elif prediction == 1:
                    result = "LEGITIMATE"
                else:
                    error = "The model returned an unexpected label."

            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "Prediction failed. Check the saved model "
                    "and feature configuration."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
        submitted_url=submitted_url,
        decision_score=decision_score
    )


if __name__ == "__main__":
    print("URLytix Flask server is starting...")
    print("Loading model from:", MODEL_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)
