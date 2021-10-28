import re
import bs4


HEADER_TAGS = ['h2', 'h3']


def extract_input_description(text):
  soup = bs4.BeautifulSoup(text, 'html.parser')
  input_header = soup.find(HEADER_TAGS, text=re.compile('Input'))
  input_text = get_text_following_header(input_header)
  return input_text


def extract_input_constraints(text):
  soup = bs4.BeautifulSoup(text, 'html.parser')
  input_header = soup.find(HEADER_TAGS, text=re.compile('Input'))
  input_text = get_text_following_header(input_header)
  constraints_header = soup.find(HEADER_TAGS, text=re.compile('Constraints'))
  constraints_text = get_text_following_header(constraints_header)
  return input_text + constraints_text


def get_text_following_header(header):
  next_node = header
  lines = []
  while next_node is not None:
    next_node = next_node.nextSibling
    if next_node is None:
      break
    elif isinstance(next_node, bs4.Tag) and next_node.name == 'h2':
      break
    elif isinstance(next_node, bs4.NavigableString):
      text = next_node.strip()
    else:
      text = next_node.get_text(strip=True).strip()
    lines.append(text)
  return '\n'.join(lines)
