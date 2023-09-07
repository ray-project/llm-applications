from bs4 import BeautifulSoup, NavigableString, Tag


def extract_text_from_section(section):
    texts = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            if elem.strip():
                texts.append(elem.strip())
        elif elem.name == "section":
            continue
        else:
            texts.append(elem.get_text().strip())
    return "\n".join(texts)


def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]


def extract_sections(record):
    with open(record["path"], "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    sections = soup.find_all("section")
    section_list = []
    for section in sections:
        section_id = section.get("id")
        section_text = extract_text_from_section(section)
        if section_id:
            uri = path_to_uri(path=record["path"])
            section_list.append({"source": f"{uri}#{section_id}", "text": section_text})
    return section_list
