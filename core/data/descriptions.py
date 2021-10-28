import bs4
import re


HEADER_TAGS = ['h2', 'h3']


def extract_input_description(text, soup=None):
  soup = soup or bs4.BeautifulSoup(text, 'html.parser')
  input_header = soup.find(HEADER_TAGS, text=re.compile('Input'))
  input_text = get_text_following_header(input_header)
  return input_text.strip()


def extract_input_constraints(text, soup=None):
  soup = soup or bs4.BeautifulSoup(text, 'html.parser')
  constraints_header = soup.find(HEADER_TAGS, text=re.compile('Constraints'))
  constraints_text = get_text_following_header(constraints_header)
  return constraints_text.strip()


def get_text_following_header(header):
  next_node = header
  lines = []
  while next_node is not None:
    next_node = next_node.nextSibling
    if next_node is None:
      break
    elif isinstance(next_node, bs4.Tag) and next_node.name in HEADER_TAGS:
      break
    elif isinstance(next_node, bs4.NavigableString):
      text = next_node.strip()
    else:
      text = next_node.get_text().strip()
    lines.append(text)
  return '\n'.join(lines)
