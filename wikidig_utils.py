import re

RE_REF_TAG = re.compile(r"<ref .+?>.+?</ref>", re.DOTALL)
RE_XML_TAG = re.compile(r"<[^>]+?>")
RE_FILE_LINK = re.compile(r"\[\[File:(.+?)(|.+?)]]")
RE_LINKS = re.compile(r"\[\[(.+?)(\|.+?)?]]")
RE_MACRO = re.compile(r"{{.+?}}")
RE_H3 = re.compile(r"^===(.+)===$", re.MULTILINE)
RE_H2 = re.compile(r"^==(.+)==$", re.MULTILINE)
RE_H1 = re.compile(r"^=(.+)=$", re.MULTILINE)
RE_EMPTY_LIST = re.compile(r"^\*\s*$", re.MULTILINE)


def remove_nested_curlies(text: str):
    # Stack to track open curly braces
    nesting_level = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '{{':  # Found an opening tag
            nesting_level += 1
            i += 2  # Skip the opening braces
        elif text[i:i+2] == '}}' and nesting_level > 0:  # Found a closing tag
            nesting_level -= 1
            i += 2  # Skip the closing braces
        else:
            if nesting_level == 0:  # Only append characters when not inside a tag
                result.append(text[i])
            i += 1
    return ''.join(result)


def remove_file_links(text: str):
    # Stack to track open curly braces
    result = []
    nesting_level = 0
    start_first_level = None
    link_content = None
    i = 0
    while i < len(text):
        if text[i:i+2] == '[[':  # Found an opening tag
            if nesting_level == 0:
                start_first_level = i+2
            nesting_level += 1
            i += 2  # Skip the opening braces
        elif text[i:i+2] == ']]' and nesting_level > 0:  # Found a closing tag
            nesting_level -= 1
            if nesting_level == 0:
                link_content = text[start_first_level:i]
                start_first_level = None
            i += 2  # Skip the closing braces
        else:
            if nesting_level == 0:  # Only append characters when not inside a tag
                result.append(text[i])
            i += 1

        if link_content:
            # special case
            result = result[0:i-len(link_content)]
            if not link_content.lower().startswith("file:"):
                p = link_content.find("|")
                if p != -1:
                    link_content = link_content[p+1:]
                result.extend(f"[{link_content}]")
            link_content = None

    return ''.join(result)


def wikimedia2md(text: str) -> str:
    """
    A simple regex-based conversion from the WikiMedia wiki format to the Markdown wiki format.
    """
    text = remove_nested_curlies(text)
    text = remove_file_links(text)

    text = RE_XML_TAG.sub("", text)
    text = RE_H3.sub(r"### \1", text)
    text = RE_H2.sub(r"## \1", text)
    text = RE_H1.sub(r"# \1", text)
    text = RE_EMPTY_LIST.sub(r"", text)

    return text.strip()
