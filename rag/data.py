from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from rag.config import EFS_DIR


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


def fetch_text(uri):
    url, anchor = uri.split("#") if "#" in uri else (uri, None)
    file_path = Path(EFS_DIR, url.split("https://")[-1])
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    if anchor:
        target_element = soup.find(id=anchor)
        if target_element:
            text = target_element.get_text()
        else:
            return fetch_text(uri=url)
    else:
        text = soup.get_text()
    return text
