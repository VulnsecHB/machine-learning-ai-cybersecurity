def rule_spam(text):
    if "free" in text.lower():
        return "spam"
    else:
        return "ham"

print(rule_spam("Win a free Iphone"))
print(rule_spam("Meeting at 10am"))
