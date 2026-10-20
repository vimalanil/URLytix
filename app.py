from flask import Flask, render_template, request
from urllib.parse import urlparse
from pathlib import Path
import ipaddress
import math
import numpy as np
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

# Readable names for the "Top contributing features" chips.
FEATURE_LABELS = {
    "URLLength": "URL length",
    "DomainLength": "Domain length",
    "TLDLength": "TLD length",
    "NoOfSubDomain": "Subdomains",
    "NoOfObfuscatedChar": "Encoded (%xx) characters",
    "ObfuscationRatio": "Encoded character ratio",
    "NoOfLettersInURL": "Letters in URL",
    "LetterRatioInURL": "Letter ratio",
    "NoOfDegitsInURL": "Digits in URL",
    "DegitRatioInURL": "Digit ratio",
    "NoOfEqualsInURL": "'=' characters",
    "NoOfQMarkInURL": "'?' characters",
    "NoOfAmpersandInURL": "'&' characters",
    "NoOfOtherSpecialCharsInURL": "Special characters",
    "SpacialCharRatioInURL": "Special character ratio",
}

# Yes/no features get a sentence instead of "name: 0/1".
BINARY_LABELS = {
    "IsDomainIP": ("Domain is an IP address", "Domain is a name, not an IP"),
    "HasObfuscation": ("Contains encoded (%xx) characters", "No encoded characters"),
    "IsHTTPS": ("Uses HTTPS", "Uses HTTP (no HTTPS)"),
}


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


def describe_url(raw_url, normalized_url):
    """Plain facts about the URL for the result card.

    Everything here is read from the URL text itself. Nothing is looked
    up or visited, so the "IP address" row only says whether the domain
    is written as an IP address, not where the domain resolves to.
    """

    parsed = urlparse(normalized_url)
    hostname = parsed.hostname or ""
    scheme_was_typed = raw_url.strip().lower().startswith(("http://", "https://"))

    try:
        ip_obj = ipaddress.ip_address(hostname)
        ip_text = f"{ip_obj} (IPv{ip_obj.version} address)"
        is_ip = True
    except ValueError:
        ip_text = "Not an IP address"
        is_ip = False

    parts = hostname.split(".")
    if is_ip or len(parts) < 2:
        tld = "—"
        subdomains = 0
    else:
        tld = "." + parts[-1]
        subdomains = max(len(parts) - 2, 0)

    protocol = parsed.scheme.lower()
    if not scheme_was_typed:
        protocol += " (assumed, no scheme entered)"

    return {
        "domain": hostname,
        "tld": tld,
        "subdomains": subdomains,
        "protocol": protocol,
        "length": len(normalized_url),
        "ip_address": ip_text,
        "is_ip": is_ip,
        "has_port": parsed.port is not None,
        "port": parsed.port,
    }


def estimate_confidence(decision_score):
    """Map the SVM margin to a 50-100 % figure.

    LinearSVC has no predict_proba, so this is a logistic squash of the
    distance from the decision boundary. It grows with the margin, but it
    is not a calibrated probability, and the page labels it "estimated".
    """

    # Capped below 100 so a large margin never reads as certainty.
    return min(round(100 / (1 + math.exp(-abs(decision_score))), 2), 99.99)


def format_value(name, value):
    if name in ("ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL",
                "SpacialCharRatioInURL"):
        return f"{value:.2f}"
    return f"{int(value)}"


def top_contributing_features(features_df, prediction, top_n=4):
    """Features that pushed the linear model hardest toward its answer.

    For a scaler + LinearSVC pipeline the decision score is
    sum(coef * scaled_value) + intercept, so each feature's share is
    coef * scaled_value. Class 1 (legitimate) is the positive side, so a
    phishing result is driven by the most negative shares.
    """

    try:
        if hasattr(model, "steps"):
            scaled = model[:-1].transform(features_df)
            classifier = model[-1]
        else:
            scaled = features_df.values
            classifier = model

        shares = np.asarray(scaled)[0] * classifier.coef_[0]
        direction = -1 if prediction == 0 else 1
        order = np.argsort(-direction * shares)

        chips = []
        for idx in order:
            if direction * shares[idx] <= 0:
                break  # nothing else pushed toward this result

            name = FEATURE_NAMES[idx]
            value = features_df.iloc[0, idx]

            if name in BINARY_LABELS:
                yes_text, no_text = BINARY_LABELS[name]
                text = yes_text if int(value) == 1 else no_text
            else:
                text = f"{FEATURE_LABELS[name]}: {format_value(name, value)}"

            chips.append(text)
            if len(chips) == top_n:
                break

        return chips

    except Exception:
        app.logger.exception("Could not compute feature contributions")
        return []


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_url = ""
    decision_score = None
    confidence = None
    url_info = None
    top_features = []

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

                if result:
                    url_info = describe_url(submitted_url, normalized_url)
                    confidence = estimate_confidence(decision_score)
                    top_features = top_contributing_features(
                        features_df, prediction
                    )

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
        decision_score=decision_score,
        confidence=confidence,
        url_info=url_info,
        top_features=top_features
    )


if __name__ == "__main__":
    print("URLytix Flask server is starting...")
    print("Loading model from:", MODEL_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)