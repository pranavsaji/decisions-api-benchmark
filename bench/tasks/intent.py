"""Customer-message intent routing on Banking77: one `choice` question with 77 options."""
from bench import metrics
from bench.tasks.base import TRAINED, Task, blank, breakdown, correct, is_probability, probability_table, rate_metric, unwrap

INTENTS = [
    "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed", "cancel_transfer",
    "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
    "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
INTENT_SET = set(INTENTS)

QUESTION = {
    "type": "choice",
    "name": "intent",
    "instructions": "Which banking intent best matches this customer message?",
    "choices": [
        {"value": intent, "description": intent.replace("_", " ").rstrip("?").capitalize() + "."}
        for intent in INTENTS
    ],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "first_choice": {"type": "string", "enum": INTENTS},
        "second_choice": {"type": "string", "enum": INTENTS},
        "third_choice": {"type": "string", "enum": INTENTS},
        "confidence": {"type": "number"},
    },
    "required": ["first_choice", "second_choice", "third_choice", "confidence"],
    "additionalProperties": False,
}

FORMAT_NOTE = (
    "Reply with JSON. first_choice, second_choice and third_choice are your three most likely "
    "options, best first. confidence is the probability, between 0 and 1, that first_choice is correct."
)


def render_input(item: dict) -> str:
    return f"Customer message: {item['text']}"


def to_prediction(endpoint: str, raw) -> dict:
    status, payload = unwrap(endpoint, raw)
    if status != "ok":
        return blank(status)
    if endpoint in TRAINED:
        table = probability_table(payload, INTENTS)
        if table is None:
            return blank("malformed")
        ordered = sorted(INTENTS, key=lambda v: -table[v])
        choice = payload.get("label") if payload.get("label") is not None else ordered[0]
        probability = table.get(choice) if isinstance(choice, str) else None
        ranked = [choice] + [v for v in ordered if v != choice][:2]
    elif endpoint == "decisions":
        if payload.get("type") != "choice":
            return blank("malformed")
        choice = payload.get("choice")
        table = {}
        entries = payload.get("probabilities")
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                table[entry["value"]] = entry.get("probability")
        probability = table.get(choice) if isinstance(choice, str) else None
        others = [v for v in table if v != choice and v in INTENT_SET and is_probability(table[v])]
        ranked = [choice] + sorted(others, key=lambda v: -table[v])[:2]
    else:
        candidates = [payload.get(key) for key in ("first_choice", "second_choice", "third_choice")]
        if not all(isinstance(c, str) and c in INTENT_SET for c in candidates):
            return blank("malformed")
        choice = candidates[0]
        probability = payload.get("confidence")
        ranked = list(dict.fromkeys(candidates))
    if not isinstance(choice, str) or choice not in INTENT_SET or not is_probability(probability):
        return blank("malformed")
    return {**blank("ok"), "label": choice, "ranked": ranked, "probability": probability}


def headline_value(pairs: list[tuple[dict, dict]]) -> float | None:
    return sum(correct(i, p) for i, p in pairs) / len(pairs) if pairs else None


def score(items: list[dict], predictions: list[dict]) -> dict:
    pairs = [(i, p) for i, p in zip(items, predictions) if p["status"] != "refused"]
    parsed = [(i, p) for i, p in pairs if p["status"] == "ok"]
    top1 = sum(correct(i, p) for i, p in pairs)
    top3 = sum(i["label"] in p["ranked"] for i, p in parsed)
    calibration = metrics.reliability(
        [p["probability"] for _, p in parsed], [int(correct(i, p)) for i, p in parsed], 0.0, 1.0, 10
    )
    return {
        "metrics": {
            "top1_accuracy": rate_metric("Top-1 accuracy", top1, len(pairs)),
            "top3_accuracy": rate_metric("Top-3 accuracy", top3, len(pairs)),
        },
        "calibration": {**calibration, "x_label": "Stated confidence", "y_label": "Share correct"},
        "breakdown": breakdown(pairs, lambda item: item["label"]),
    }


TASK = Task(
    name="intent",
    title="Intent routing",
    fields=[("text", "Customer message")],
    questions=[QUESTION],
    responses_schema=SCHEMA,
    format_note=FORMAT_NOTE,
    render_input=render_input,
    to_prediction=to_prediction,
    score=score,
    headline="top1_accuracy",
    headline_value=headline_value,
    examples=[
        {"text": "I ordered my card two weeks ago and it still has not turned up."},
        {"text": "Why was I charged a fee when I took cash out abroad?"},
        {"text": "Someone has my phone and I am worried they can get into the app."},
    ],
)
