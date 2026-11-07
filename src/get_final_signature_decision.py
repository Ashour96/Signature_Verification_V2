from collections import Counter


def get_final_signature_decision(results, min_agreement=0.70):

    decisions = [item["decision"] for item in results]

    votes = Counter(decisions)

    total = len(decisions)

    genuine_votes = votes.get("GENUINE", 0)
    forgery_votes = votes.get("FORGERY", 0)

    majority_decision = votes.most_common(1)[0][0]

    agreement = max(genuine_votes, forgery_votes) / total

    if agreement < min_agreement:
        final_decision = "REVIEW"
    else:
        final_decision = majority_decision

    return {
        "final_decision": final_decision,
        "majority_decision": majority_decision,
        "genuine_votes": genuine_votes,
        "forgery_votes": forgery_votes,
        "agreement_percentage": round(agreement * 100, 2),
        "distances_mean": sum(
            item["distance"] for item in results
        ) / total
    }