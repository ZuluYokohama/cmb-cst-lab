"""Parse complete pytest result lines in saved evidence logs."""
import re


def parse_test_summary(text):
    counts = r"\d+ (?:passed|skipped|failed|errors?|warnings?|xfailed|xpassed|deselected)"
    pattern = rf"({counts}(?:, {counts})*) in \d+(?:\.\d+)?s(?: \([^\n]*\))?"
    for line in reversed(text.splitlines()):
        match = re.fullmatch(pattern, line.strip().strip("=").strip())
        if match:
            result = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0}
            for item in match[1].split(", "):
                count, label = item.split()
                result["errors" if label == "error" else label] = int(count)
            return result
    return None
