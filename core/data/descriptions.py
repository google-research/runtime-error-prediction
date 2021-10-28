import bs4
import fire
import re


HEADER_TAGS = ['h2', 'h3']
INPUT_HEADER_NAMES = ['Input', '入力']
CONSTRAINT_HEADER_NAMES = ['Constraints', '制約', '入力形式']
# 入力形式 = Input format.


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


def extract_input_information(text, soup=None):
  soup = soup or bs4.BeautifulSoup(text, 'html.parser')
  info = []
  for header_name in INPUT_HEADER_NAMES + CONSTRAINT_HEADER_NAMES:
    header = soup.find(HEADER_TAGS, text=re.compile(header_name))
    text = get_text_following_header(header)
    if text:
      info.append((header_name, text))
  return '\n\n'.join(f'{header}:\n{text}')


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


def get_all_input_descriptions():
  for i in range(4200):
    path = f'/mnt/project-codenet-storage/Project_CodeNet/problem_descriptions/p{i:05d}.html'
    with open(path, 'r') as f:
      text = f.read()
    info = extract_input_information(text)
    # input_description = extract_input_description(text)
    # input_constraints = extract_input_constraints(text)
    print('Index:', i)
    print(info)
    print()
    # print('Description:', input_description)
    # print('Constraints:', input_constraints)
    # print()

if __name__ == '__main__':
  fire.Fire()
